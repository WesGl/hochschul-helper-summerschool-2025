# Hochschul-Helper Summer School 2025
Welcome to Hochschul-Helper, an AI-powered assistant built to help students navigate the Summer School 2025 program. This application uses advanced RAG (Retrieval-Augmented Generation) technology to provide accurate and contextual answers to your questions about courses, schedules, professors, and more.

## Features
- **Natural Language Queries**: Ask questions in your own words
- **Course Information**: Get details about available courses, prerequisites, and credits
- **Schedule Assistance**: Find when and where classes are held
- **Professor Information**: Learn about instructors and their expertise
- **Application Guidance**: Get help with application processes and deadlines
- **PDF Document Retrieval**: Access information stored in PDF documents through the RAG system
- **Online HKA Information Search**: Search for up-to-date information about Hochschule Karlsruhe
- **Calendar Event Planning**: Generate ICS files for HKA events or your personal HKA semester timetable

## How to Use
1. **Setup venv**: install uv package manager and run `uv sync`
2. **Start the Application**: Run `uv run chainlit run app.py` in your terminal
3. **Ask Questions**: Type your question in the chat interface
4. **Get Answers**: The system will provide relevant information from the Summer School 2025 database

## Technical Components
- **Chainlit**: Provides the chat interface
- **Vector Database**: Stores embedded documents for efficient retrieval
- **LangChain**: Orchestrates the RAG workflow
- **LLM**: Powers the understanding and generation of responses
- **Multi-Agent System**: Coordinates between RAG retrieval, web search, and calendar functionality

## Getting Started
```bash
# Install dependencies
uv sync

# Set up your environment variables
export OPENAI_API_KEY=your_api_key_here

# Run the application
uv run chainlit run app.py
```

## Customization
You can customize the knowledge base by adding or modifying documents in the `data/` directory. The system will automatically process new information during startup.

## Support
If you encounter any issues or have questions about the implementation, please open an issue in the GitHub repository.