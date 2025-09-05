<!-- markdownlint-disable first-line-h1 no-inline-html -->

> [!NOTE]
> This repository is publicly accessible as part of our open-source initiative. We welcome contributions from the community alongside our organization's primary development efforts.

---

# cloudetair-chatbot-api

The <font color="#2f4b94">CLOUDET<font color="#c6c6c6"><i>ai</i></font>R</font> chatbot API provides the backend services for the [cloudetair-chatbot-frontend](https://github.com/cloudeteer/cloudetair-chatbot-frontend).

## Features

- Builds a containerized API service.
- Pushes the image to GitHub Container Registry.
- Fully automated via GitHub Actions workflows.
- Serves as a reusable base for further customizations.

## Development

```shell
# Build the container image
docker compose build

# Start the containers in detached mode
docker compose up --detach

# The api will be available at http://localhost:8000

# Stop containers and remove volumes
docker compose down --volumes
```
