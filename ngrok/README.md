# Using ngrok for local development

## Dashboard URL
[ngrok Dashboard](https://dashboard.ngrok.com)
- your login is just using google acount

## Install and Configure
You've already done this, but if you need to again here it is.
```zsh
brew install ngrok
```

You can easily find `YOUR_TOKEN` in the ngrok dashboard.
```zsh
ngrok config add-authtoken $YOUR_TOKEN
```

Then just run the below command using whatever `PORT` you are currently using, with FastAPI it's more than likely port `8000`.
```zsh
ngrok http $PORT
```

Or a better way is to configure an ngrok `endpoints`.

To do this you need to add an endpoint to the `ngrok.yml` file like so. You can edit the file directly or run `ngrok config edit` at the terminal and it will allow you to edit in the terminal.

```yaml
version: 3
agent:
  authtoken: $YOUR_TOKEN
endpoints:
  - name: fastapi-ngrok-tunnel
    url: $YOUR_DOMAIN
    upstream:
      url: 8000

```

Then you can just run the below command.

```zsh
ngrok start fastapi-ngrok-tunnel
```

That's about it.
