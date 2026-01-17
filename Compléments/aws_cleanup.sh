#!/bin/bash

set -euo pipefail

echo "===== Suppression des groupes de sécurité non par défaut ====="
for sg in $(aws ec2 describe-security-groups \
    --query "SecurityGroups[?GroupName!='default'].GroupId" \
    --output text); do
    echo "🗑️ Suppression du groupe de sécurité : $sg"
    aws ec2 delete-security-group --group-id "$sg" || echo "❌ Échec suppression SG $sg (probablement encore utilisé)"
done

echo "===== Suppression des politiques IAM personnalisées ====="
for policy_arn in $(aws iam list-policies --scope Local \
    --query "Policies[].Arn" --output text); do
    echo "🗑️ Suppression de la politique : $policy_arn"
    aws iam delete-policy --policy-arn "$policy_arn" || echo "❌ Échec suppression politique $policy_arn"
done

echo "===== Suppression des profils d’instance (avant les rôles) ====="
for profile in $(aws iam list-instance-profiles \
    --query "InstanceProfiles[].InstanceProfileName" --output text); do
    echo "🔍 Traitement du profil : $profile"

    for role in $(aws iam get-instance-profile --instance-profile-name "$profile" \
        --query "InstanceProfile.Roles[].RoleName" --output text); do
        echo "⛓️ Détachement du rôle $role du profil $profile"
        aws iam remove-role-from-instance-profile \
            --instance-profile-name "$profile" \
            --role-name "$role" || echo "❌ Échec détachement rôle $role"
    done

    echo "🗑️ Suppression du profil : $profile"
    aws iam delete-instance-profile --instance-profile-name "$profile" || echo "❌ Échec suppression profil $profile"
done

echo "===== Suppression des rôles IAM personnalisés ====="
for role in $(aws iam list-roles \
    --query "Roles[?starts_with(RoleName, 'AWSServiceRoleFor') == \`false\` && starts_with(RoleName, 'AWSReservedSSO_') == \`false\`].RoleName" \
    --output text); do
    echo "🔍 Traitement du rôle : $role"

    echo "⛓️ Détachement des politiques managées..."
    for policy_arn in $(aws iam list-attached-role-policies --role-name "$role" \
        --query "AttachedPolicies[].PolicyArn" --output text); do
        aws iam detach-role-policy --role-name "$role" --policy-arn "$policy_arn" || echo "❌ Échec détachement policy $policy_arn"
    done

    echo "🧾 Suppression des politiques inline..."
    for inline in $(aws iam list-role-policies --role-name "$role" \
        --query "PolicyNames[]" --output text); do
        aws iam delete-role-policy --role-name "$role" --policy-name "$inline" || echo "❌ Échec suppression inline $inline"
    done

    echo "🗑️ Suppression du rôle : $role"
    aws iam delete-role --role-name "$role" || echo "❌ Échec suppression rôle $role"
done

echo "✅ Nettoyage terminé avec succès."
