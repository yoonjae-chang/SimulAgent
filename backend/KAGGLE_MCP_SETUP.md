# Kaggle MCP Server Setup

This project is configured to use the Kaggle MCP server from [54yyyu/kaggle-mcp](https://github.com/54yyyu/kaggle-mcp).

## Installation

The kaggle-mcp package has already been installed in the virtual environment:

```bash
# Already done in .venv
pip install git+https://github.com/54yyyu/kaggle-mcp.git
```

## Configuration

### 1. Get Kaggle API Credentials

1. Go to [Kaggle Settings](https://www.kaggle.com/settings)
2. Scroll to the "API" section
3. Click "Create New Token"
4. This will download a `kaggle.json` file containing your credentials

### 2. Set Up Environment Variables

The `kaggle.json` file contains your credentials in this format:

```json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_key"
}
```

Copy `.env.example` to `.env` and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add:

```
DEDALUS_API_KEY=your_dedalus_api_key

KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

### 3. Optional: Traditional Kaggle Setup

Alternatively, you can place the `kaggle.json` file in the standard location:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

If you use this method, the environment variables in `.env` are optional as the Kaggle library will automatically read from `~/.kaggle/kaggle.json`.

## Usage

The Kaggle MCP server is configured in `main.py` and will be automatically available when running your Dedalus Labs agent. It provides access to:

- Kaggle datasets
- Kaggle competitions
- Dataset search and download
- Competition leaderboards and submissions

## Testing

To test the configuration, run:

```bash
cd backend
source .venv/bin/activate
python main.py
```

The agent should now have access to Kaggle functionality through the MCP server.

## Troubleshooting

### Authentication Errors

If you see authentication errors:
- Verify your credentials in `.env` match your Kaggle account
- Check that `KAGGLE_USERNAME` and `KAGGLE_KEY` are correctly set
- Ensure the `kaggle.json` file (if using) has correct permissions (600)

### Module Not Found

If you see "module not found" errors:
- Ensure you're using the virtual environment: `source .venv/bin/activate`
- Verify kaggle-mcp is installed: `pip list | grep kaggle-mcp`
- Reinstall if needed: `pip install git+https://github.com/54yyyu/kaggle-mcp.git`

## Available Tools

The Kaggle MCP server provides the following tools to your agent:

- `search_datasets`: Search for Kaggle datasets
- `download_dataset`: Download a specific dataset
- `list_competitions`: List available competitions
- `get_competition_leaderboard`: Get competition standings
- `authenticate`: Manually authenticate with Kaggle credentials

Refer to the [kaggle-mcp repository](https://github.com/54yyyu/kaggle-mcp) for full documentation.
