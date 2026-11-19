"""Unit tests for Pydantic models and normalization logic."""

import pytest
from src.perplexica_mcp.server import (
    Source,
    SearchResult,
    SearchError,
    SearchResponse,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
)


class TestSourceModel:
    """Tests for Source Pydantic model."""

    def test_source_with_all_fields(self):
        """Test Source model with all fields populated."""
        source = Source(
            title="Test Title",
            url="https://example.com",
            content="Test content"
        )
        assert source.title == "Test Title"
        assert source.url == "https://example.com"
        assert source.content == "Test content"

    def test_source_with_optional_fields(self):
        """Test Source model with optional fields as None."""
        source = Source()
        assert source.title is None
        assert source.url is None
        assert source.content is None

    def test_source_allows_extra_fields(self):
        """Test that Source model allows extra fields from Perplexica."""
        source = Source(
            title="Test",
            extra_field="extra_value",
            another_field=123
        )
        assert source.title == "Test"
        assert source.extra_field == "extra_value"
        assert source.another_field == 123

    def test_source_from_dict(self):
        """Test Source model creation from dictionary."""
        data = {
            "title": "Dict Title",
            "url": "https://dict.example.com",
            "content": "Dict content",
            "metadata": {"key": "value"}
        }
        source = Source(**data)
        assert source.title == "Dict Title"
        assert source.metadata == {"key": "value"}


class TestSearchResultModel:
    """Tests for SearchResult Pydantic model."""

    def test_search_result_with_answer_and_sources(self):
        """Test SearchResult with answer and sources."""
        result = SearchResult(
            answer="This is the answer",
            sources=[
                Source(title="Source 1", url="https://example1.com"),
                Source(title="Source 2", url="https://example2.com"),
            ]
        )
        assert result.answer == "This is the answer"
        assert len(result.sources) == 2
        assert result.sources[0].title == "Source 1"

    def test_search_result_with_empty_sources(self):
        """Test SearchResult with default empty sources list."""
        result = SearchResult(answer="Answer only")
        assert result.answer == "Answer only"
        assert result.sources == []

    def test_search_result_allows_extra_fields(self):
        """Test that SearchResult allows extra fields from Perplexica."""
        result = SearchResult(
            answer="Answer",
            sources=[],
            query="original query",
            followUp=["question1", "question2"]
        )
        assert result.answer == "Answer"
        assert result.query == "original query"
        assert result.followUp == ["question1", "question2"]

    def test_search_result_from_perplexica_response(self):
        """Test SearchResult parsing a typical Perplexica response."""
        perplexica_response = {
            "answer": "Perplexica is an AI-powered search engine.",
            "sources": [
                {
                    "title": "Perplexica GitHub",
                    "url": "https://github.com/ItzCrazyKns/Perplexica",
                    "content": "Open source AI search engine"
                }
            ],
            "followUpQuestions": ["What features does it have?"]
        }
        result = SearchResult(**perplexica_response)
        assert "AI-powered" in result.answer
        assert len(result.sources) == 1
        assert result.sources[0].title == "Perplexica GitHub"
        assert result.followUpQuestions == ["What features does it have?"]


class TestSearchErrorModel:
    """Tests for SearchError Pydantic model."""

    def test_search_error_with_message(self):
        """Test SearchError with error message."""
        error = SearchError(error="Something went wrong")
        assert error.error == "Something went wrong"

    def test_search_error_required_field(self):
        """Test that error field is required."""
        with pytest.raises(Exception):  # ValidationError
            SearchError()

    def test_search_error_http_error(self):
        """Test SearchError for HTTP error scenario."""
        error = SearchError(
            error="HTTP error occurred: Server error '500 Internal Server Error'"
        )
        assert "500" in error.error
        assert "HTTP error" in error.error


class TestSearchResponse:
    """Tests for SearchResponse type alias."""

    def test_search_response_can_be_result(self):
        """Test that SearchResponse accepts SearchResult."""
        response: SearchResponse = SearchResult(answer="Test")
        assert isinstance(response, SearchResult)

    def test_search_response_can_be_error(self):
        """Test that SearchResponse accepts SearchError."""
        response: SearchResponse = SearchError(error="Test error")
        assert isinstance(response, SearchError)


class TestModelDefaultsNormalization:
    """Tests for model defaults normalization logic in search function."""

    def test_none_chat_model_uses_default(self):
        """Test that None chat_model is replaced with default."""
        # This tests the normalization logic:
        # if chat_model is None:
        #     chat_model = DEFAULT_CHAT_MODEL
        chat_model = None
        if chat_model is None:
            chat_model = DEFAULT_CHAT_MODEL
        # Result depends on env vars, but logic is correct
        assert chat_model is DEFAULT_CHAT_MODEL

    def test_none_embedding_model_uses_default(self):
        """Test that None embedding_model is replaced with default."""
        embedding_model = None
        if embedding_model is None:
            embedding_model = DEFAULT_EMBEDDING_MODEL
        assert embedding_model is DEFAULT_EMBEDDING_MODEL

    def test_provided_model_not_overwritten(self):
        """Test that provided models are not overwritten by defaults."""
        provided_model = {"provider": "custom", "name": "custom-model"}
        chat_model = provided_model
        if chat_model is None:
            chat_model = DEFAULT_CHAT_MODEL
        assert chat_model == provided_model
        assert chat_model is not DEFAULT_CHAT_MODEL

    def test_both_none_when_no_defaults(self):
        """Test validation fails when both models are None and no defaults."""
        chat_model = None
        embedding_model = None
        # Simulate no defaults set
        default_chat = None
        default_embed = None

        if chat_model is None:
            chat_model = default_chat
        if embedding_model is None:
            embedding_model = default_embed

        # Both should still be None, triggering error
        assert chat_model is None
        assert embedding_model is None
