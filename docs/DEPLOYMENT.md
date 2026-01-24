# Deployment Guide

This guide covers deploying GM Agent with Foundry VTT, including communication modes and configuration for various hosting scenarios.

## Communication Modes

GM Agent supports two communication modes with Foundry VTT:

### WebSocket Mode (Default)

In WebSocket mode, Foundry VTT connects TO gm-agent via Socket.IO:

```
┌─────────────────┐  WebSocket   ┌──────────────────┐
│  Foundry VTT    │─────────────►│    gm-agent      │
│  (client)       │◄─────────────│    (server)      │
└─────────────────┘              └──────────────────┘
```

**Best for:**
- Foundry and gm-agent on the same machine
- Foundry and gm-agent on the same local network
- Direct network access between Foundry and gm-agent

**Configuration:**
- Default mode, no special configuration needed
- Set `wsUrl` in Foundry module settings to gm-agent server URL

### Polling Mode

In polling mode, gm-agent polls Foundry VTT via REST API:

```
┌─────────────────┐   HTTP(S)    ┌──────────────────┐
│  Foundry VTT    │◄─────────────│    gm-agent      │
│  (server)       │─────────────►│    (client)      │
└─────────────────┘              └──────────────────┘
```

**Best for:**
- Foundry hosted on cloud services (The Forge, Molten Hosting, etc.)
- Foundry behind firewalls or NAT that block incoming WebSocket connections
- Foundry behind reverse proxies that don't support WebSocket upgrades
- Any scenario where gm-agent cannot receive incoming connections

**Configuration (gm-agent):**
```bash
# Set communication mode to polling
FOUNDRY_MODE=polling

# Foundry VTT URL (must be accessible from gm-agent)
FOUNDRY_POLL_URL=https://your-foundry.example.com

# API key for authentication (must match Foundry module setting)
FOUNDRY_API_KEY=your-secure-api-key-here

# Campaign ID to connect to
FOUNDRY_CAMPAIGN_ID=my-campaign

# Optional: Polling interval in seconds (default: 2.0)
FOUNDRY_POLL_INTERVAL=2.0

# Optional: Long-poll timeout in seconds (default: 25.0)
FOUNDRY_LONG_POLL_TIMEOUT=25.0

# Optional: Verify SSL certificates (default: true)
FOUNDRY_VERIFY_SSL=true
```

**Configuration (Foundry Module):**
1. Open Module Settings → GM Agent
2. Enable "Enable Polling API"
3. Set "Polling API Key" to a secure random key (32+ characters)
4. The same API key must be set in gm-agent's `FOUNDRY_API_KEY`

## Choosing a Mode

| Scenario | Recommended Mode |
|----------|------------------|
| Foundry and gm-agent on same machine | WebSocket |
| Foundry on LAN, gm-agent on same LAN | WebSocket |
| Foundry on cloud hosting (Forge, etc.) | Polling |
| Foundry behind corporate firewall | Polling |
| Foundry behind reverse proxy without WebSocket | Polling |
| Foundry accessible via VPN | Either (WebSocket preferred) |

## Security Considerations

### API Key Security

When using polling mode:

1. **Generate a strong API key**: Use at least 32 random characters
   ```bash
   # Generate a secure API key
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Keep the API key secret**: Only share between Foundry module and gm-agent

3. **Use HTTPS in production**: Set `FOUNDRY_VERIFY_SSL=true` (default)

### Network Security

- **WebSocket mode**: gm-agent needs to accept incoming connections (port 5000 by default)
- **Polling mode**: Foundry needs to accept incoming HTTPS connections
- Consider using a VPN or private network when possible
- Use firewalls to restrict access to trusted IPs

### Rate Limiting

The Foundry module includes basic rate limiting:
- Maximum 1000 events stored in queue
- Events expire after 5 minutes
- Recommended poll interval: 2 seconds

## Deployment Scenarios

### Local Development

Both Foundry and gm-agent on the same machine:

```bash
# Terminal 1: Start gm-agent
cd gm-agent
gm server

# Foundry VTT runs on localhost:30000
# gm-agent runs on localhost:5000
# Configure Foundry module wsUrl: http://localhost:5000
```

### Docker Deployment (Same Network)

```yaml
# docker-compose.yml
version: '3.8'
services:
  gm-agent:
    build: ./gm-agent
    ports:
      - "5000:5000"
    environment:
      - FOUNDRY_MODE=websocket
```

Configure Foundry module `wsUrl` to `http://gm-agent:5000` (internal Docker network) or `http://host-ip:5000` (external access).

### Cloud Foundry + Local gm-agent (Polling Mode)

When Foundry is hosted on a cloud service:

1. **Foundry (Cloud)**:
   - Enable "Enable Polling API" in module settings
   - Set a secure API key
   - Ensure HTTPS is enabled

2. **gm-agent (Local)**:
   ```bash
   export FOUNDRY_MODE=polling
   export FOUNDRY_POLL_URL=https://your-forge-game.forge-vtt.com
   export FOUNDRY_API_KEY=your-api-key
   export FOUNDRY_CAMPAIGN_ID=my-campaign
   gm server
   ```

### Cloud Foundry + Cloud gm-agent (Polling Mode)

Both services in the cloud:

1. **gm-agent server** (e.g., on a VPS):
   ```bash
   # .env
   FOUNDRY_MODE=polling
   FOUNDRY_POLL_URL=https://your-foundry.com
   FOUNDRY_API_KEY=secure-key
   FOUNDRY_CAMPAIGN_ID=my-campaign
   ```

2. Use a process manager (systemd, PM2) or container orchestration

## Troubleshooting

### WebSocket Mode Issues

**Problem**: Foundry can't connect to gm-agent
- Check firewall allows incoming connections on port 5000
- Verify gm-agent is running: `curl http://localhost:5000/api/health`
- Check Foundry module logs for connection errors
- Ensure correct `wsUrl` in Foundry settings

**Problem**: Connection drops frequently
- Check network stability
- Increase Socket.IO reconnection attempts in module settings
- Check for proxy/load balancer timeout settings

### Polling Mode Issues

**Problem**: gm-agent can't reach Foundry
- Verify Foundry URL is accessible: `curl https://your-foundry.com`
- Check SSL certificate validity if using HTTPS
- Ensure "Enable Polling API" is enabled in Foundry

**Problem**: Authentication errors (401)
- Verify API key matches in both Foundry module and gm-agent `.env`
- Check for extra whitespace in API key configuration

**Problem**: Events not being received
- Check Foundry module logs for polling API initialization
- Verify automation is enabled in Foundry
- Check event queue status in Foundry module

### General Issues

**Problem**: Automation not responding
- Check gm-agent logs for errors
- Verify campaign exists: `gm campaign list`
- Check bridge connection status via API: `GET /api/campaigns/{id}/automation`

## API Endpoints

### gm-agent Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/campaigns/{id}/automation` | GET | Get automation status |
| `/api/campaigns/{id}/automation` | POST | Toggle automation |

### Foundry Polling API Endpoints

When polling mode is enabled, Foundry exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/gm-agent/status` | GET | API status and health |
| `/api/gm-agent/events` | GET | Poll for events |
| `/api/gm-agent/command` | POST | Execute command |

All endpoints require `Authorization: Bearer <api_key>` header.

## Performance Tuning

### Polling Interval

Adjust `FOUNDRY_POLL_INTERVAL` based on your needs:
- Lower (1.0s): More responsive, higher server load
- Higher (5.0s): Less responsive, lower server load
- Default (2.0s): Good balance for most use cases

### Long-Poll Timeout

The `FOUNDRY_LONG_POLL_TIMEOUT` setting enables long-polling:
- Server holds request open until events arrive or timeout
- Reduces polling frequency when idle
- Default 25.0s works well with most proxies/firewalls

### Event Queue Size

The Foundry module's event queue is configured for:
- Maximum 1000 events
- 5-minute event expiration
- Events are pruned automatically

For high-activity games, these limits prevent memory issues while ensuring recent events are captured.
