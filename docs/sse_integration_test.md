# SSE Integration Test

## Auto Accepted Case
```
curl --location 'http://localhost:8000/api/chat/stream' \
--header 'Accept: */*' \
--header 'Accept-Language: en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7' \
--header 'Cache-Control: no-cache' \
--header 'Connection: keep-alive' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "user",
            "content": "what is mcp?"
        }
    ],
    "thread_id": "test_thread_1",
    "auto_accepted_plan": true
}'
```

## Human in the Loop Case

### Initial Plan
```
curl --location 'http://localhost:8000/api/chat/stream' \
--header 'Accept: */*' \
--header 'Accept-Language: en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7' \
--header 'Cache-Control: no-cache' \
--header 'Connection: keep-alive' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "user",
            "content": "what is mcp?"
        }
    ],
    "thread_id": "test_thread_2",
    "auto_accepted_plan": false
}'
```


### Edit the plan
```
{
    "messages": [
        {
            "role": "user",
            "content": "make the last step be comprehensive"
        }
    ],
    "thread_id": "test_thread_2",
    "auto_accepted_plan": false,
    "interrupt_feedback": "edit_plan"
}
```

### Accept the plan
```
{
    "thread_id": "test_thread_2",
    "auto_accepted_plan": false,
    "interrupt_feedback": "accepted"
}
```