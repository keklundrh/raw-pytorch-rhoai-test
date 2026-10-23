# Test raw pytorch model deployments 

## Repository structure 

* [`local`](./local) - code to auth huggingface token and test CPU model deployments 
* [`openshift-ai`](./openshift-ai) - code to deploy models to OpenShift AI, see [README.md](./openshift-ai/README.md)
* `tests` - basic testing, incomplete 

## Note on repo 

Models are custom raw PyTorch deployments which means they will not, by default, deploy using OpenShift AI default runtimes. The options are: 
- use  a custom serving runtime _(what we tested)_
- convert to ONNX format for use with OpenVINO or MLServer _(not yet tested)_ 

## What's next 

- Determine which ONNX port makes sense 
- Add cuda support for model throughput performance 
- Consider ModelCar pattern for easier deployment management
