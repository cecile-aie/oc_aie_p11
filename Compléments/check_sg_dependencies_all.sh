#!/bin/bash

declare -A SG_LIST
SG_LIST["eu-west-1"]="sg-07b3cd0246ef44b5a sg-02578897e5e8d3a72"
SG_LIST["us-east-1"]="sg-03f3c79c46bc56bf1 sg-007476defcd394be5"

for REGION in "${!SG_LIST[@]}"; do
  echo "=================================================="
  echo "🔎 Région : $REGION"
  echo "=================================================="

  for SG_ID in ${SG_LIST[$REGION]}; do
    echo "--------------------------------------------------"
    echo "🔍 Analyse des dépendances pour : $SG_ID"
    echo "--------------------------------------------------"

    echo "➡️ Interfaces réseau (ENI)..."
    aws ec2 describe-network-interfaces --region "$REGION" \
      --filters Name=group-id,Values="$SG_ID" \
      --query "NetworkInterfaces[*].{ID:NetworkInterfaceId,Status:Status,Attachment:Attachment.InstanceId,Desc:Description}" \
      --output table

    echo "➡️ Instances EC2..."
    aws ec2 describe-instances --region "$REGION" \
      --filters Name=instance.group-id,Values="$SG_ID" \
      --query "Reservations[*].Instances[*].{ID:InstanceId,State:State.Name}" \
      --output table

    echo "➡️ Load Balancers (ELBv2)..."
    aws elbv2 describe-load-balancers --region "$REGION" \
      --query "LoadBalancers[?SecurityGroups!=null && contains(SecurityGroups, '$SG_ID')]" \
      --output table

    echo "➡️ Launch Templates..."
    for tmpl in $(aws ec2 describe-launch-templates --region "$REGION" \
        --query "LaunchTemplates[].LaunchTemplateId" --output text); do
      aws ec2 describe-launch-template-versions --region "$REGION" \
        --launch-template-id "$tmpl" \
        --query "LaunchTemplateVersions[?LaunchTemplateData.NetworkInterfaces[?Groups[?GroupId=='$SG_ID']]]" \
        --output table
    done

    echo "➡️ Auto Scaling Groups..."
    aws autoscaling describe-auto-scaling-groups --region "$REGION" \
      --query "AutoScalingGroups[?contains(SecurityGroups, '$SG_ID')]" \
      --output table

    echo "➡️ Classic ELB..."
    aws elb describe-load-balancers --region "$REGION" \
      --query "LoadBalancerDescriptions[?contains(SecurityGroups, '$SG_ID')]" \
      --output table

    echo "➡️ Fin de l'analyse pour $SG_ID"
    echo
  done
done
