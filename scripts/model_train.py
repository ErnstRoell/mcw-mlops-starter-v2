import model
import click

@click.command()
@click.option('--model-name', type=str, help='bla')
@click.option('--build-number', type=str, help='bla')
def run_training(model_name,build_number):
    model.train(['--model-name',model_name,
                 '--build-number',build_number])
run_training()
