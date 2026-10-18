#!/usr/bin/env bash
# Run the evaluation on the existing GPU box (g5.xlarge, A10G 24 GB, i-02b17b44a3cb0641f, us-west-2; the 250 GB
# disk holds the HF model cache and a ~/venv with CUDA torch + transformers from the earlier experiments).
# Needs a live AWS session: `aws login` first.
#   scripts/aws.sh start                 # start the stopped box, record its IP, open SSH from this IP
#   scripts/aws.sh setup                 # rsync the repo, pip install -e it into ~/venv
#   scripts/aws.sh run NAME CMD...       # rsync + run detached on the box; log in ~/rescomp/logs/NAME.log
#   scripts/aws.sh log NAME [N]          # tail a log
#   scripts/aws.sh ssh CMD...            # anything else
#   scripts/aws.sh pull                  # results/ (jsonl + generations, not the .rsc files) back to ./results/
#   scripts/aws.sh stop                  # stop (not terminate): no compute charge, disk kept
#   scripts/aws.sh cap MINUTES           # in-box shutdown timer (stop) as a spending cap
# A second, short-lived box (BOX=big): `BOX=big ITYPE=g6e.xlarge scripts/aws.sh launch` creates it with
# instance-initiated shutdown = terminate and a BUDGET_MIN timer; the other commands then address it with BOX=big.
#   BOX=big scripts/aws.sh terminate
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
export AWS_REGION=${AWS_REGION:-us-west-2}
BOX=${BOX:-main}
STATE=$HERE/scripts/.box$([ "$BOX" = main ] || echo "-$BOX")
if [ "$BOX" = main ]; then ID=${BOX_ID:-i-02b17b44a3cb0641f}; else ID=$(cat "$STATE.id" 2>/dev/null || true); fi
KEY=$HOME/.ssh/sae-cpu.pem
ip() { cat "$STATE.ip"; }
SSH() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 ubuntu@"$(ip)" "$@"; }
push() { rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" --exclude .venv --exclude results --include 'tests/data/*.rsc' --exclude '*.rsc' \
           --exclude __pycache__ --exclude .pytest_cache --exclude 'scripts/.box*' "$HERE/" "ubuntu@$(ip):rescomp/"; }
cmd=$1; shift
case $cmd in
  start)
    aws ec2 start-instances --instance-ids "$ID" >/dev/null
    aws ec2 wait instance-running --instance-ids "$ID"
    aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text > "$STATE.ip"
    SG=$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)
    MYIP=$(curl -s https://checkip.amazonaws.com)
    aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true
    echo "box $ID up at $(ip)"
    until SSH true 2>/dev/null; do sleep 5; done
    SSH "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h / | tail -1; ~/venv/bin/python -c 'import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)'" ;;
  setup)
    push
    SSH "~/venv/bin/pip install -q -e ~/rescomp zstandard datasets pytest accelerate 2>&1 | tail -2; cd ~/rescomp && ~/venv/bin/python -m pytest 2>&1 | tail -1" ;;
  run)
    push; name=$1; shift
    SSH "mkdir -p ~/rescomp/logs && cd ~/rescomp && nohup ~/venv/bin/python $(printf '%q ' "$@") > logs/$name.log 2>&1 < /dev/null &"
    echo "started $name" ;;
  log)  SSH "tail -n ${2:-40} ~/rescomp/logs/$1.log" ;;
  push) push ;;
  ssh)  SSH "$@" ;;
  pull) mkdir -p "$HERE/results"
        rsync -az -e "ssh -i $KEY" --exclude '*.rsc' "ubuntu@$(ip):rescomp/results/" "$HERE/results/"
        rsync -az -e "ssh -i $KEY" "ubuntu@$(ip):rescomp/logs/" "$HERE/results/logs/" ;;
  cap)  SSH "sudo shutdown -h +$1 'rescomp spending cap'"; echo "box shuts itself down in $1 min" ;;
  launch)
    [ "$BOX" != main ] || { echo "set BOX=<name> for a new box"; exit 1; }
    [ -z "$ID" ] || { echo "$STATE.id exists ($ID); terminate it first"; exit 1; }
    SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=sae-cpu --query 'SecurityGroups[0].GroupId' --output text)
    AMI=$(aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id --query Parameter.Value --output text)
    MYIP=$(curl -s https://checkip.amazonaws.com)
    aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true
    ID=""
    for SUBNET in $(aws ec2 describe-subnets --filters Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text); do
      ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "${ITYPE:-g6e.xlarge}" --key-name sae-cpu --security-group-ids "$SG" \
        --subnet-id "$SUBNET" --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${DISK_GB:-200},VolumeType=gp3,DeleteOnTermination=true}" \
        --user-data file://$HERE/scripts/bootstrap.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=rescomp-$BOX}]" \
        --query 'Instances[0].InstanceId' --output text 2>/dev/null) && break || echo "no capacity in $SUBNET"
    done
    [ -n "$ID" ] || { echo "no capacity in any AZ for ${ITYPE:-g6e.xlarge}"; exit 1; }
    echo "$ID" > "$STATE.id"
    aws ec2 wait instance-running --instance-ids "$ID"
    aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text > "$STATE.ip"
    echo "launched $ID (${ITYPE:-g6e.xlarge}) at $(ip)"
    until SSH test -f .bootstrap-done 2>/dev/null; do sleep 15; done
    SSH "sudo shutdown -h +${BUDGET_MIN:-240} 'rescomp budget cap'" && echo "terminates itself in ${BUDGET_MIN:-240} min"
    SSH "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; ~/venv/bin/python -c 'import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)'" ;;
  terminate)
    [ "$BOX" != main ] || { echo "refusing to terminate the main box; use stop"; exit 1; }
    aws ec2 terminate-instances --instance-ids "$ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text && rm -f "$STATE.id" "$STATE.ip" ;;
  stop) [ "$BOX" = main ] || { echo "a launched box is terminate-on-shutdown; use terminate"; exit 1; }
        aws ec2 stop-instances --instance-ids "$ID" --query 'StoppingInstances[0].CurrentState.Name' --output text ;;
  status) aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress,InstanceType]' --output text ;;
  *) echo "unknown command $cmd"; exit 1 ;;
esac
