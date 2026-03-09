# Exercise 1.6 - Docker DokuWiki

This folder replicates the Docker + DokuWiki workflow from class.

You can validate that the committed scaffold is present with:

```bash
python exercise_1/main.py --exercise 1_6
```

## 1. Start Docker

- Linux (systemd):
  `sudo systemctl start docker.service`
- Windows/macOS:
  start Docker Desktop and wait for "Engine running".

## 2. Start the container

From this folder (`exercise_1/step_6/`):

```bash
docker compose up -d
```

If `dokuwiki_data/` does not exist yet, Docker will create it on first run.

Open:
`http://localhost:8080/`

## 3. Check process and logs

```bash
docker ps
docker logs dokuwiki-lab
docker exec -it dokuwiki-lab bash
```

## 4. Check persistent files

```bash
ls dokuwiki_data
```

On Windows PowerShell:

```powershell
Get-ChildItem .\dokuwiki_data
```

## 5. Stop and remove the container

```bash
docker compose down
```

## 6. List local Docker images

```bash
docker image ls
```

Equivalent one-shot command (as shown in slides):

```bash
docker run -d -p 8080:8080 --user 1000:1000 -v ./dokuwiki_data:/storage --name dokuwiki-lab dokuwiki/dokuwiki:stable
```
