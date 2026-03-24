#https://docs.ollama.com/integrations/claude-code

curl -fsSL https://claude.ai/install.sh | bash

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

ollama launch claude 
## glm-4.7-flash  19GB
minimax-m2


ollama launch claude --model kimi-k2.5:cloud


