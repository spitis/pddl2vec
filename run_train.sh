#!/bin/bash
srun --gres=gpu:1 -p gpu --mem=64G python value_iteration.py
