### single restart
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-instant-1.2 --n-retry 1 --verbose --only-system-plus-assistant
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.0 --n-retry 1 --verbose --only-system-plus-assistant
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.1 --n-retry 1 --verbose --only-system-plus-assistant
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-haiku-20240307 --n-retry 1 --verbose
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-sonnet-20240229 --n-retry 1 --verbose --only-system-plus-assistant
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-opus-20240229 --n-retry 1 --verbose --only-system-plus-assistant


### more restarts
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-instant-1.2       --n-retry 10 --verbose --only-system-plus-assistant
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.0               --n-retry 10 --verbose --only-system-plus-assistant
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.1               --n-retry 10 --verbose --only-system-plus-assistant
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-haiku-20240307   --n-retry 10 --verbose --only-system-plus-assistant
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-sonnet-20240229 --n-retry 10 --verbose --only-system-plus-assistant
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-opus-20240229   --n-retry 10 --verbose --only-system-plus-assistant
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.1   --n-retry 100 --verbose --only-system-plus-assistant


### further ablation study: what's the most minimal attack?
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-instant-1.2 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt ''
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-instant-1.2 --n-retry 1 --verbose --only-system-plus-assistant --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-instant-1.2 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-instant-1.2 --n-retry 10 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'

python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.0 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt ''
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.0 --n-retry 1 --verbose --only-system-plus-assistant --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.0 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.0 --n-retry 10 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'

python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.1 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt ''
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.1 --n-retry 1 --verbose --only-system-plus-assistant --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.1 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-2.1 --n-retry 10 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'

python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-haiku-20240307 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt ''
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-haiku-20240307 --n-retry 1 --verbose --only-system-plus-assistant --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-haiku-20240307 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-haiku-20240307 --n-retry 10 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'

python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-sonnet-20240229 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt ''
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-sonnet-20240229 --n-retry 1 --verbose --only-system-plus-assistant --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-sonnet-20240229 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-sonnet-20240229 --n-retry 10 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'

python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-opus-20240229 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt ''
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-opus-20240229 --n-retry 1 --verbose --only-system-plus-assistant --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-opus-20240229 --n-retry 1 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'
python main_claude_prefilling.py --n-behaviors 50 --target-model claude-3-opus-20240229   --n-retry 10 --verbose --only-system-plus-assistant --system-prompt '' --user-prompt '-'
