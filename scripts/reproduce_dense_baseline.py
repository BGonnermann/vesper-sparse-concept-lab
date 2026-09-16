"""Run the frozen TinyStories dense baseline into a NEW artifact directory."""
import argparse
from pathlib import Path
import foundation_baseline as baseline

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True,help='New campaign-artifact root; never overwrite an earlier attempt')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    baseline.HOME=args.output.resolve()
    baseline.main()
