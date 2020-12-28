import azureml.core
from azureml.core import Environment
from azureml.core.workspace import Workspace
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
import numpy as np
from azureml.core.authentication import AzureCliAuthentication
import argparse

parser = argparse.ArgumentParser("create_aml_cluster")
parser.add_argument("--path", type=str, help="path", dest="path", required=True)
args = parser.parse_args()

print("Argument 1: %s" % args.path)

print('creating AzureCliAuthentication...')
cli_auth = AzureCliAuthentication()
print('done creating AzureCliAuthentication!')

print('get workspace...')
ws = Workspace.from_config(path=args.path, auth=cli_auth)
print('done getting workspace!')


print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')
experiment_name = 'train-on-local'
exp = Experiment(workspace=ws, name=experiment_name)



# Editing a run configuration property on-fly.
user_managed_env = Environment("user-managed-env")
user_managed_env.python.user_managed_dependencies = True
src = ScriptRunConfig(source_directory='./scripts/', 
                    script='train.py', 
                    environment=user_managed_env)





run = exp.submit(src)
run.wait_for_completion(show_output=True)

# 6.B System-managed environment

system_managed_env = Environment("system-managed-env")
system_managed_env.python.user_managed_dependencies = False

# Specify conda dependencies with scikit-learn
cd = CondaDependencies.create(conda_packages=['scikit-learn'])
system_managed_env.python.conda_dependencies = cd

src.run_config.environment = system_managed_env
run = exp.submit(src)
run.wait_for_completion(show_output=True)

run.get_metrics()
metrics = run.get_metrics()


best_alpha = metrics['alpha'][np.argmin(metrics['mse'])]

print('When alpha is {1:0.2f}, we have min MSE {0:0.2f}.'.format(
    min(metrics['mse']), 
    best_alpha
))



#  [markdown]
# From the results obtained above, `ridge_0.40.pkl` is the best performing model. You can now register that particular model with the workspace. Once you have done so, go back to the portal and click on "Models". You should see it there.

# 
# Supply a model name, and the full path to the serialized model file.
model = run.register_model(model_name='best_ridge_model', model_path='./outputs/ridge_0.40.pkl')

print("Registered model:\n --> Name: {}\n --> Version: {}\n --> URL: {}".format(model.name, model.version, model.url))

