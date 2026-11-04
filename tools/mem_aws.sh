#!/usr/bin/env bash
# Start / stop / check the personal g5.xlarge used for experiments.
# Uses the AWS CLI profile `mem-aws` (temporary credentials; regenerate when expired)
# and keeps `Host mem-aws` in ~/.ssh/config pointed at the current public DNS.
#
#   tools/mem_aws.sh status | start | stop
#   tools/mem_aws.sh push    # rsync the working tree to ~/mem-aware (no .git; writes REVISION)
#   tools/mem_aws.sh pull    # rsync ~/mem-aware/results back into ./results
set -euo pipefail
ID=i-08f69883b92cd545c
AWS=(aws --profile mem-aws --region us-west-2)
cd "$(dirname "$0")/.."

dns() {
  "${AWS[@]}" ec2 describe-instances --instance-ids "$ID" \
    --query 'Reservations[0].Instances[0].[State.Name,PublicDnsName]' --output text
}

case "${1:-status}" in
  status) dns ;;
  start)
    "${AWS[@]}" ec2 start-instances --instance-ids "$ID" >/dev/null
    "${AWS[@]}" ec2 wait instance-running --instance-ids "$ID"
    host=$(dns | awk '{print $2}')
    # Rewrite only the HostName line inside the `Host mem-aws` block.
    sed -i '' "/^Host mem-aws$/,/^Host /s/^\( *HostName \).*/\1$host/" ~/.ssh/config
    echo "running: $host (ssh mem-aws)"
    ;;
  stop)
    "${AWS[@]}" ec2 stop-instances --instance-ids "$ID" >/dev/null
    echo "stopping $ID"
    ;;
  push)
    rev=$(git rev-parse HEAD 2>/dev/null || echo unknown)
    git diff --quiet HEAD 2>/dev/null || rev="$rev+dirty"
    echo "$rev" > REVISION
    rsync -az --exclude .git --exclude data --exclude models --exclude results --exclude .venv \
      --exclude __pycache__ ./ mem-aws:~/mem-aware/
    rm -f REVISION
    echo "pushed $rev"
    ;;
  pull)
    rsync -az mem-aws:~/mem-aware/results/ results/
    echo "pulled into results/"
    ;;
  *) echo "usage: $0 status|start|stop|push|pull" >&2; exit 2 ;;
esac
