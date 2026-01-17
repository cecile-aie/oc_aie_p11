#!/bin/bash

set -euo pipefail

# Liste des régions activées sur le compte
regions=$(aws ec2 describe-regions --query "Regions[].RegionName" --output text)

for region in $regions; do
  echo "======================"
  echo "🔁 Région : $region"
  echo "======================"

  echo "🧹 Suppression des groupes de sécurité (hors default)..."
  for sg in $(aws ec2 describe-security-groups --region "$region" \
      --query "SecurityGroups[?GroupName!='default'].GroupId" \
      --output text); do
    aws ec2 delete-security-group --region "$region" --group-id "$sg" \
      && echo "✅ SG supprimé : $sg" \
      || echo "❌ Échec SG : $sg"
  done

  echo "🧹 Suppression des ENI disponibles..."
  for eni in $(aws ec2 describe-network-interfaces --region "$region" \
      --query "NetworkInterfaces[?Status=='available'].NetworkInterfaceId" \
      --output text); do
    aws ec2 delete-network-interface --region "$region" --network-interface-id "$eni" \
      && echo "✅ ENI supprimé : $eni" \
      || echo "❌ Échec ENI : $eni"
  done

  echo "🧹 Suppression des instances EC2..."
  for instance in $(aws ec2 describe-instances --region "$region" \
      --query "Reservations[*].Instances[*].InstanceId" --output text); do
    aws ec2 terminate-instances --region "$region" --instance-ids "$instance" \
      && echo "🛑 EC2 terminée : $instance"
  done

  echo "🧹 Suppression des clusters EMR..."
  for cluster_id in $(aws emr list-clusters --region "$region" \
      --cluster-states STARTING BOOTSTRAPPING RUNNING WAITING \
      --query "Clusters[].Id" --output text); do
    aws emr terminate-clusters --region "$region" --cluster-ids "$cluster_id"
    echo "🛑 Cluster EMR résilié : $cluster_id"
  done
done

echo "======================"
echo "🧼 Nettoyage global IAM (hors régions)"
echo "======================"

echo "🧹 Suppression des politiques personnalisées..."
for policy_arn in $(aws iam list-policies --scope Local \
    --query "Policies[].Arn" --output text); do
  aws iam delete-policy --policy-arn "$policy_arn" || echo "❌ Politique non supprimée : $policy_arn"
done

echo "🧹 Suppression des profils d’instance..."
for profile in $(aws iam list-instance-profiles \
    --query "InstanceProfiles[].InstanceProfileName" --output text); do
  for role in $(aws iam get-instance-profile --instance-profile-name "$profile" \
      --query "InstanceProfile.Roles[].RoleName" --output text); do
    aws iam remove-role-from-instance-profile \
      --instance-profile-name "$profile" --role-name "$role" || true
  done
  aws iam delete-instance-profile --instance-profile-name "$profile" || echo "❌ Échec profil $profile"
done

echo "🧹 Suppression des rôles IAM personnalisés..."
for role in $(aws iam list-roles \
    --query "Roles[?starts_with(RoleName, 'AWSServiceRoleFor') == \`false\` && starts_with(RoleName, 'AWSReservedSSO_') == \`false\`].RoleName" \
    --output text); do
  for policy_arn in $(aws iam list-attached-role-policies --role-name "$role" \
      --query "AttachedPolicies[].PolicyArn" --output text); do
    aws iam detach-role-policy --role-name "$role" --policy-arn "$policy_arn" || true
  done
  for inline in $(aws iam list-role-policies --role-name "$role" \
      --query "PolicyNames[]" --output text); do
    aws iam delete-role-policy --role-name "$role" --policy-name "$inline" || true
  done
  aws iam delete-role --role-name "$role" || echo "❌ Échec suppression rôle $role"
done

echo "✅ Réinitialisation complète (sauf S3) terminée."
