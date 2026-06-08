curl -X POST "https://llm-gateway-colo01-colo.zoomdev.us/v1/chat/completions" \
  -H "X-Api-Key: $APIKEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt_oss_120b",
    "messages": [
      {
        "role": "user",
        "content": "Who are you?"
      }
    ]
  }'