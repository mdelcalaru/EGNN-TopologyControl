import yaml
import subprocess
import sys
from pathlib import Path

def read_wandb_sweep_info(yaml_path):
    """
    Lee el archivo YAML del sweep para obtener entity y project.
    """
    with open(yaml_path, "r") as f:
        sweep_config = yaml.safe_load(f)
    entity = sweep_config.get("entity", None)
    project = sweep_config.get("project", None)
    return entity, project

def run_sweep(yaml_path="sweep/sweep_config.yaml"):
    print("Starting wandb sweep...")
    result = subprocess.run(
        ["wandb", "sweep", yaml_path],
        capture_output=True,
        text=True,
        check=True
    )
    output = result.stdout + result.stderr
    print(output)
    sweep_id = None
    for line in output.splitlines():
        if "Creating sweep with ID" in line:
            sweep_id = line.split()[-1]
            break
    if sweep_id is None:
        print("Error: sweep ID not found in wandb output.")
        sys.exit(1)
    print(f"Sweep ID detected: {sweep_id}")
    return sweep_id

def run_agent(sweep_id, entity, project, num_agents=1):
    full_sweep_id = f"{entity}/{project}/{sweep_id}"
    try:
        for i in range(num_agents):
            print(f"Starting agent {i+1}/{num_agents} for sweep {full_sweep_id}...")
            proc = subprocess.Popen(
                ["wandb", "agent", full_sweep_id],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            proc.wait()
    except KeyboardInterrupt:
        print("Interrupted, terminating current agent...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Runner for wandb sweeps")
    parser.add_argument("--start_sweep", action="store_true", help="Start a new wandb sweep and get its sweep_id")
    parser.add_argument("--run_agents", action="store_true", help="Run sweep agents after starting sweep")
    parser.add_argument("--num_agents", type=int, default=1, help="Number of sweep agents to run")
    parser.add_argument("--sweep_id", type=str, default=None, help="Existing sweep id to run agents for")
    parser.add_argument("--yaml_path", type=str, default="sweep/sweep_config.yaml", help="Path to sweep YAML file")
    args = parser.parse_args()

    entity, project = read_wandb_sweep_info(args.yaml_path)
    if entity is None or project is None:
        print("Error: 'entity' or 'project' not found in sweep YAML.")
        sys.exit(1)

    if args.start_sweep:
        sweep_id = run_sweep(args.yaml_path)
        if args.run_agents:
            run_agent(sweep_id, entity, project, args.num_agents)
    elif args.run_agents:
        if args.sweep_id is None:
            print("Error: --sweep_id is required to run agents without starting sweep.")
            sys.exit(1)
        run_agent(args.sweep_id, entity, project, args.num_agents)
    else:
        print("Specify --start_sweep or --run_agents")
