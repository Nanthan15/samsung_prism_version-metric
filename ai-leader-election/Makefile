# Define variables
DOCKER_IMAGE_NAME = nanthan26/ai-leader-election
DOCKER_IMAGE_TAG = latest
NAMESPACE = ai-leader
K8S_DEPLOYMENT_NAME = ai-leader-app

build-image:
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) -f app/Dockerfile .



# Push the Docker image to the registry
push-image:
	docker push $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

# Start Minikube
start-minikube:
	minikube start

# Set the Kubernetes namespace
set-namespace:
	kubectl create namespace $(NAMESPACE) || true
	kubectl config set-context --current --namespace=$(NAMESPACE)

# Run the AI model in Kubernetes (e.g., retrain, collect metrics)
run-model:
	kubectl run ai-leader-model --image=$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) --restart=Never --namespace=$(NAMESPACE) --command -- python /app/model.py

# Deploy Kubernetes resources (deployment, services, cronjobs, etc.)
deploy-k8s:
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/cronjob.yaml
	kubectl apply -f k8s/service.yaml

# Check Kubernetes pods in the ai-leader namespace
get-pods:
	kubectl get pods -n $(NAMESPACE)

# Get the status of Kubernetes jobs
get-jobs:
	kubectl get jobs -n $(NAMESPACE)

# Get logs of a specific pod (replace with your pod name)
get-pod-logs:
	kubectl logs ai-leader-app-7874d67b-lq2sg -n $(NAMESPACE)

# Delete the Kubernetes resources (e.g., deployment, cronjobs, services)
delete-k8s:
	kubectl delete -f k8s/deployment.yaml
	kubectl delete -f k8s/cronjob.yaml
	kubectl delete -f k8s/service.yaml

# Clean up Docker images (remove locally built images)
clean-docker:
	docker rmi $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

# Delete all pods in the ai-leader namespace
delete-pods:
	kubectl delete pods --all -n $(NAMESPACE)

# Rebuild Docker image and deploy to Kubernetes
rebuild-deploy:
	$(MAKE) build-image
	$(MAKE) push-image
	$(MAKE) deploy-k8s

# Default target to run all necessary steps
all: build-image push-image start-minikube set-namespace deploy-k8s run-model
