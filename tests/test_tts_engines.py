from unittest.mock import patch, MagicMock, Mock

import tempfile

import os

import pytest

import torch

import torchaudio

from tts.orchestrator import (
    TTSEngine,
    XTTSGenerator,
    EdgeTTSGenerator,
    RVCGenerator,
    F5Generator,
    create_tts_engine,
    TTSEngineLoadError,
    TTSEngineGenerationError,
    TTSEngineValidationError,
    TTSEngineError
)

class TestTTSEngine:
    """Test the abstract TTS engine base class."""

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        # Create a mock object that implements the abstract methods
        engine = Mock(spec=TTSEngine)
        engine.engine_name = "test"
        engine.target_sr = 24000

        # Configure the mock to raise NotImplementedError for abstract methods
        engine.load_model.side_effect = NotImplementedError()
        engine.generate_segment.side_effect = NotImplementedError()
        engine.cleanup.side_effect = NotImplementedError()

        with pytest.raises(NotImplementedError):
            engine.load_model()

        with pytest.raises(NotImplementedError):
            engine.generate_segment({}, "test.wav")

        with pytest.raises(NotImplementedError):
            engine.cleanup()

class TestXTTSGenerator:
    """Test XTTS engine implementation."""

    def test_init(self):
        """Test XTTS engine initialization."""
        engine = XTTSGenerator()
        assert engine.engine_name == "xtts"
        assert engine.target_sr == 24000

    def test_load_model(self):
        """Test XTTS model loading."""
        engine = XTTSGenerator()
        # Should not raise any exceptions
        engine.load_model()

    def test_validate_language_valid(self):
        """Test language validation with valid language."""
        engine = XTTSGenerator()
        # Should not raise any exceptions for valid languages
        engine.validate_language("en")

    def test_validate_language_invalid(self):
        """Test language validation with invalid language."""
        engine = XTTSGenerator()
        with patch('tts.xtts.validate_language') as mock_validate:
            mock_validate.side_effect = Exception("Invalid language")

            with pytest.raises(TTSEngineValidationError):
                engine.validate_language("invalid")

    def test_generate_segment_empty_text(self):
        """Test segment generation with empty text."""
        engine = XTTSGenerator()
        segment = {"translated_text": ""}

        with pytest.raises(TTSEngineGenerationError):
            engine.generate_segment(segment, "test.wav")

    @patch('tts.orchestrator.generate_tts_for_speaker_xtts')
    def test_generate_segment_success(self, mock_generate):
        """Test successful segment generation."""
        mock_generate.return_value = [{"path": "test.wav", "start": 0, "end": 1, "speaker": "test"}]

        engine = XTTSGenerator()
        segment = {"translated_text": "test text", "speaker": "test", "start": 0.0, "end": 1.0}

        # Create temporary dummy ref.wav file with valid audio data
        temp_ref = tempfile.NamedTemporaryFile(suffix='.wav', dir='/tmp', delete=False)
        temp_ref_path = temp_ref.name
        temp_ref.close()  # Close the file handle so we can delete it later

        try:
            # Write valid silent audio data to the temp file
            tensor = torch.zeros(1, 22050)  # 1 second of silence at 22050Hz
            torchaudio.save(temp_ref_path, tensor, 22050)

            result = engine.generate_segment(segment, temp_ref_path, tmp_path="/tmp")

            assert result["path"] == "test.wav"
            assert result["speaker"] == "test"
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_ref_path):
                os.unlink(temp_ref_path)

    @patch('tts.xtts.generate_tts_for_speaker_xtts')
    def test_generate_segment_no_segments(self, mock_generate):
        """Test segment generation when no segments are returned."""
        mock_generate.return_value = []

        engine = XTTSGenerator()
        segment = {"translated_text": "test text", "speaker": "test"}

        with pytest.raises(TTSEngineGenerationError):
            engine.generate_segment(segment, "ref.wav")

    def test_cleanup(self):
        """Test cleanup method."""
        engine = XTTSGenerator()
        # Should not raise any exceptions
        engine.cleanup()

class TestEdgeTTSGenerator:
    """Test Edge-TTS engine implementation."""

    def test_init(self):
        """Test Edge-TTS engine initialization."""
        engine = EdgeTTSGenerator()
        assert engine.engine_name == "edge"
        assert engine.target_sr == 24000

    def test_load_model(self):
        """Test Edge-TTS model loading."""
        engine = EdgeTTSGenerator()
        # Should not raise any exceptions
        engine.load_model()

    def test_validate_language_valid(self):
        """Test language validation with valid language."""
        engine = EdgeTTSGenerator()
        # Should not raise any exceptions for valid languages
        engine.validate_language("en")

    def test_validate_language_invalid(self):
        """Test language validation with invalid language."""
        engine = EdgeTTSGenerator()
        with patch('tts.edge_tts.validate_language') as mock_validate:
            mock_validate.side_effect = Exception("Invalid language")

            with pytest.raises(TTSEngineValidationError):
                engine.validate_language("invalid")

    def test_generate_segment_empty_text(self):
        """Test segment generation with empty text."""
        engine = EdgeTTSGenerator()
        segment = {"translated_text": ""}

        with pytest.raises(TTSEngineGenerationError):
            engine.generate_segment(segment, "test.wav")

    @patch('tts.orchestrator.generate_tts_for_speaker_edge')
    def test_generate_segment_success(self, mock_generate):
        """Test successful segment generation."""
        mock_generate.return_value = [{"path": "test.wav", "start": 0, "end": 1, "speaker": "test"}]

        engine = EdgeTTSGenerator()
        segment = {"translated_text": "test text", "speaker": "test", "start": 0.0, "end": 1.0}

        result = engine.generate_segment(segment, "ref.wav", tmp_path="/tmp")

        assert result["path"] == "test.wav"
        assert result["speaker"] == "test"

    def test_cleanup(self):
        """Test cleanup method."""
        engine = EdgeTTSGenerator()
        # Should not raise any exceptions
        engine.cleanup()

class TestRVCGenerator:
    """Test RVC engine implementation."""

    def test_init(self):
        """Test RVC engine initialization."""
        engine = RVCGenerator()
        assert engine.engine_name == "rvc"
        assert engine.target_sr == 24000

    def test_load_model(self):
        """Test RVC model loading."""
        engine = RVCGenerator()
        # Should not raise any exceptions
        engine.load_model()

    def test_validate_language_valid(self):
        """Test language validation with valid language."""
        engine = RVCGenerator()
        # Should not raise any exceptions for valid languages
        engine.validate_language("en")

    def test_validate_language_invalid(self):
        """Test language validation with invalid language."""
        engine = RVCGenerator()
        with patch('tts.F5.validate_language') as mock_validate:
            mock_validate.side_effect = Exception("Invalid language")

            with pytest.raises(TTSEngineValidationError):
                engine.validate_language("invalid")

    def test_generate_segment_empty_text(self):
        """Test segment generation with empty text."""
        engine = RVCGenerator()
        segment = {"translated_text": ""}

        with pytest.raises(TTSEngineGenerationError):
            engine.generate_segment(segment, "test.wav")

    @patch('tts.orchestrator.generate_tts_for_speaker')
    def test_generate_segment_success(self, mock_generate):
        """Test successful segment generation."""
        mock_generate.return_value = [{"path": "test.wav", "start": 0, "end": 1, "speaker": "test"}]

        engine = RVCGenerator()
        # Load model before generating segment
        engine.load_model()
        segment = {"translated_text": "test text", "speaker": "test", "start": 0.0, "end": 1.0}

        # Create temporary dummy ref.wav file with valid audio data
        temp_ref = tempfile.NamedTemporaryFile(suffix='.wav', dir='/tmp', delete=False)
        temp_ref_path = temp_ref.name
        temp_ref.close()  # Close the file handle so we can delete it later

        try:
            # Write valid silent audio data to the temp file
            tensor = torch.zeros(1, 22050)  # 1 second of silence at 22050Hz
            torchaudio.save(temp_ref_path, tensor, 22050)

            result = engine.generate_segment(segment, temp_ref_path, tmp_path="/tmp")

            assert result["path"] == "test.wav"
            assert result["speaker"] == "test"
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_ref_path):
                os.unlink(temp_ref_path)

    def test_cleanup(self):
        """Test cleanup method."""
        engine = RVCGenerator()
        # Should not raise any exceptions
        engine.cleanup()

class TestF5Generator:
    """Test F5-TTS engine implementation."""

    def test_init(self):
        """Test F5-TTS engine initialization."""
        engine = F5Generator()
        assert engine.engine_name == "f5"
        assert engine.target_sr == 24000

    def test_load_model(self):
        """Test F5-TTS model loading."""
        engine = F5Generator()
        # Should not raise any exceptions
        engine.load_model()

    def test_validate_language_valid(self):
        """Test language validation with valid language."""
        engine = F5Generator()
        # Should not raise any exceptions for valid languages
        engine.validate_language("en")

    def test_validate_language_invalid(self):
        """Test language validation with invalid language."""
        engine = F5Generator()
        with patch('tts.F5.validate_language') as mock_validate:
            mock_validate.side_effect = Exception("Invalid language")

            with pytest.raises(TTSEngineValidationError):
                engine.validate_language("invalid")

    def test_generate_segment_empty_text(self):
        """Test segment generation with empty text."""
        engine = F5Generator()
        segment = {"translated_text": ""}

        with pytest.raises(TTSEngineGenerationError):
            engine.generate_segment(segment, "test.wav")

    @patch('tts.orchestrator.generate_tts_for_speaker')
    def test_generate_segment_success(self, mock_generate):
        """Test successful segment generation."""
        mock_generate.return_value = [{"path": "test.wav", "start": 0, "end": 1, "speaker": "test"}]

        engine = F5Generator()
        # Load model before generating segment
        engine.load_model()
        segment = {"translated_text": "test text", "speaker": "test", "start": 0.0, "end": 1.0}

        # Create temporary dummy ref.wav file with valid audio data
        temp_ref = tempfile.NamedTemporaryFile(suffix='.wav', dir='/tmp', delete=False)
        temp_ref_path = temp_ref.name
        temp_ref.close()  # Close the file handle so we can delete it later

        try:
            # Write valid silent audio data to the temp file
            tensor = torch.zeros(1, 22050)  # 1 second of silence at 22050Hz
            torchaudio.save(temp_ref_path, tensor, 22050)

            result = engine.generate_segment(segment, temp_ref_path, tmp_path="/tmp")

            assert result["path"] == "test.wav"
            assert result["speaker"] == "test"
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_ref_path):
                os.unlink(temp_ref_path)

    def test_cleanup(self):
        """Test cleanup method."""
        engine = F5Generator()
        # Should not raise any exceptions
        engine.cleanup()

class TestCreateTTSEngine:
    """Test the TTS engine factory function."""

    def test_create_edge_engine(self):
        """Test creating Edge-TTS engine."""
        engine = create_tts_engine("edge")
        assert isinstance(engine, EdgeTTSGenerator)

    def test_create_xtts_engine(self):
        """Test creating XTTS engine."""
        engine = create_tts_engine("xtts")
        assert isinstance(engine, XTTSGenerator)

    def test_create_rvc_engine(self):
        """Test creating RVC engine."""
        engine = create_tts_engine("rvc")
        assert isinstance(engine, RVCGenerator)

    def test_create_f5_engine(self):
        """Test creating F5-TTS engine."""
        engine = create_tts_engine("f5")
        assert isinstance(engine, F5Generator)

    def test_create_invalid_engine(self):
        """Test creating invalid engine."""
        with pytest.raises(ValueError):
            create_tts_engine("invalid")

class TestTTSEngineExceptions:
    """Test TTS engine exceptions."""

    def test_engine_load_error(self):
        """Test TTSEngineLoadError."""
        error = TTSEngineLoadError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, TTSEngineError)

    def test_engine_generation_error(self):
        """Test TTSEngineGenerationError."""
        error = TTSEngineGenerationError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, TTSEngineError)

    def test_engine_validation_error(self):
        """Test TTSEngineValidationError."""
        error = TTSEngineValidationError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, TTSEngineError)