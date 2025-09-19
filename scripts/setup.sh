cd ./terraform
terraform init
terraform apply -auto-approve
cd ..

# Wait for 5 seconds
sleep 5

cd ./ansible
ansible-playbook site.yml
