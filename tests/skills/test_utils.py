"""
Unit tests for skills utility functions
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from xagent.skills.utils import (
    _get_default_skill_dirs,
    _parse_skill_dirs,
    create_skill_manager,
)


class TestParseSkillDirs:
    """Tests for _parse_skill_dirs function"""

    def test_empty_string(self):
        """Test empty string returns empty list"""
        result = _parse_skill_dirs("")
        assert result == []

    def test_single_valid_directory(self):
        """Test single valid directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _parse_skill_dirs(tmpdir)
            assert len(result) == 1
            assert result[0] == Path(tmpdir)

    def test_multiple_valid_directories(self):
        """Test multiple valid directories"""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                result = _parse_skill_dirs(f"{tmpdir1},{tmpdir2}")
                assert len(result) == 2
                assert result[0] == Path(tmpdir1)
                assert result[1] == Path(tmpdir2)

    def test_nonexistent_directory(self):
        """Test nonexistent directory is still added (may be created later)"""
        result = _parse_skill_dirs("/nonexistent/path")
        assert len(result) == 1
        assert result[0] == Path("/nonexistent/path")

    def test_mixed_valid_invalid(self):
        """Test mix of valid and invalid directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _parse_skill_dirs(f"{tmpdir},/nonexistent,/another/nonexistent")
            # All paths are added (nonexistent dirs may be created later)
            assert len(result) == 3
            assert result[0] == Path(tmpdir)
            assert result[1] == Path("/nonexistent")
            assert result[2] == Path("/another/nonexistent")

    def test_path_expansion_tilde(self):
        """Test tilde expansion"""
        result = _parse_skill_dirs("~/test")
        # Path is expanded and added even if it doesn't exist yet
        assert len(result) == 1
        assert result[0] == Path.home() / "test"

    def test_path_expansion_environment_variable(self):
        """Test environment variable expansion"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["TEST_SKILLS_DIR"] = tmpdir
            try:
                result = _parse_skill_dirs("$TEST_SKILLS_DIR")
                assert len(result) == 1
                assert result[0] == Path(tmpdir)
            finally:
                del os.environ["TEST_SKILLS_DIR"]

    def test_whitespace_handling(self):
        """Test whitespace handling in paths"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _parse_skill_dirs(f"  {tmpdir}  ,  {tmpdir}  ")
            assert len(result) == 2

    def test_empty_components(self):
        """Test empty components are skipped"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _parse_skill_dirs(f"{tmpdir},,{tmpdir}")
            assert len(result) == 2

    def test_url_paths_rejected(self):
        """Test URL-like paths are rejected"""
        result = _parse_skill_dirs("s3://bucket/skills,nfs://server/skills")
        assert result == []  # URLs should be rejected

    def test_file_path_accepted(self):
        """Test file paths are accepted (may be replaced with directory later)"""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile_path = tmpfile.name
        try:
            result = _parse_skill_dirs(tmpfile_path)
            # File paths are now accepted (admin may delete file and create dir)
            assert len(result) == 1
            assert result[0] == Path(tmpfile_path)
        finally:
            # Clean up the file
            try:
                os.unlink(tmpfile_path)
            except PermissionError:
                pass  # File might be locked on Windows


class TestGetDefaultSkillDirs:
    """Tests for _get_default_skill_dirs function"""

    def test_returns_three_directories(self):
        """Test returns exactly three directories"""
        result = _get_default_skill_dirs()
        assert len(result) == 3

    def test_builtin_directory_exists(self):
        """Test builtin directory exists"""
        result = _get_default_skill_dirs()
        assert result[0].exists()

    def test_project_directory_is_relative(self):
        """Test project directory is relative path './skills/'"""
        result = _get_default_skill_dirs()
        assert result[1] == Path("skills")

    def test_user_directory_path(self):
        """Test user directory is ~/.xagent/skills"""
        result = _get_default_skill_dirs()
        assert result[2].name == "skills"
        # Parent should be .xagent (or whatever get_storage_root returns)


class TestCreateSkillManager:
    """Tests for create_skill_manager function"""

    def test_no_environment_variable(self):
        """Test default behavior without environment variable"""
        with patch.dict(os.environ, {}, clear=True):
            manager = create_skill_manager()
            assert manager is not None
            # Should use default directories (builtin, project, user)
            assert len(manager.skills_roots) == 3

    def test_with_environment_variable(self):
        """Test with XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS set"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ, {"XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS": tmpdir}
            ):
                manager = create_skill_manager()
                assert manager is not None
                # Should have 3 default dirs + 1 external dir
                assert len(manager.skills_roots) == 4
                # Last one should be the external dir
                assert manager.skills_roots[3] == Path(tmpdir)

    def test_with_multiple_directories(self):
        """Test with multiple directories in environment variable"""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                env_value = f"{tmpdir1},{tmpdir2}"
                with patch.dict(
                    os.environ, {"XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS": env_value}
                ):
                    manager = create_skill_manager()
                    assert manager is not None
                    # Should have 3 default dirs + 2 external dirs
                    assert len(manager.skills_roots) == 5
                    # Last two should be the external dirs
                    assert manager.skills_roots[3] == Path(tmpdir1)
                    assert manager.skills_roots[4] == Path(tmpdir2)

    def test_with_invalid_environment_variable_falls_back_to_default(self):
        """Test external paths are added even if they don't exist yet"""
        with patch.dict(
            os.environ, {"XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS": "/nonexistent/path"}
        ):
            manager = create_skill_manager()
            assert manager is not None
            # Should have 3 default dirs + 1 external dir (nonexistent dirs are still added)
            assert len(manager.skills_roots) == 4
            assert manager.skills_roots[3] == Path("/nonexistent/path")

    def test_explicit_skills_roots_parameter(self):
        """Test explicit skills_roots parameter is used and env var is appended"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ, {"XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS": "/other/path"}
            ):
                manager = create_skill_manager(skills_roots=[Path(tmpdir)])
                assert manager is not None
                # User-provided skills_roots + env var
                assert len(manager.skills_roots) == 2
                assert manager.skills_roots[0] == Path(tmpdir)
                assert manager.skills_roots[1] == Path("/other/path")

    def test_url_paths_in_environment_variable(self):
        """Test URL paths in environment variable are skipped"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_value = f"s3://bucket/skills,{tmpdir},nfs://server/skills"
            with patch.dict(
                os.environ, {"XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS": env_value}
            ):
                manager = create_skill_manager()
                assert manager is not None
                # Should have 3 default dirs + 1 valid external dir (URLs skipped)
                assert len(manager.skills_roots) == 4
                assert manager.skills_roots[3] == Path(tmpdir)

    def test_path_expansion_in_environment_variable(self):
        """Test path expansion works in environment variable"""
        home = Path.home()
        test_dir = home / "test_xagent_skills"
        try:
            test_dir.mkdir(exist_ok=True)
            with patch.dict(
                os.environ,
                {"XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS": "~/test_xagent_skills"},
            ):
                manager = create_skill_manager()
                assert manager is not None
                # Should have 3 default dirs + 1 external dir
                assert len(manager.skills_roots) == 4
                assert manager.skills_roots[3] == test_dir
        finally:
            test_dir.rmdir()


@pytest.mark.integration
class TestSkillManagerIntegration:
    """Integration tests for skill manager with environment variables"""

    def test_end_to_end_with_real_skills(self):
        """Test end-to-end flow with real skill directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test skill
            skill_dir = Path(tmpdir) / "test_skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# Test Skill\n\nA test skill.")

            # Set environment variable
            with patch.dict(
                os.environ, {"XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS": tmpdir}
            ):
                manager = create_skill_manager()
                import asyncio

                asyncio.run(manager.initialize())

                # Verify skills were loaded (builtin + external)
                skills = asyncio.run(manager.list_skills())
                # Should have builtin skills + our test skill
                assert len(skills) >= 1
                # Check that our test skill is there
                test_skill_found = any(s["name"] == "test_skill" for s in skills)
                assert test_skill_found, (
                    "Test skill should be loaded from external directory"
                )

    def test_default_behavior_without_config(self):
        """Test default behavior when no configuration is provided"""
        with patch.dict(os.environ, {}, clear=True):
            manager = create_skill_manager()
            import asyncio

            asyncio.run(manager.initialize())

            # Should load builtin skills
            skills = asyncio.run(manager.list_skills())
            # At minimum, should have some builtin skills (evidence-based-rag, poster-design, presentation-generator)
            assert len(skills) >= 3
