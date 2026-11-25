"""Тесты преобразования MCP JSON Schema в FunctionDeclaration Gemini."""

import unittest

from google.genai import types

from mcp_client import MCPToolInfo
from mcp_schema import sanitize_schema, to_gemini_declaration, unique_public_name

READ_TEXT_FILE_SCHEMA = {
    "type": "object",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "properties": {
        "path": {
            "type": "string",
            "description": "The file to read as text.",
        },
        "tail": {
            "description": "Last N lines",
            "type": "integer",
            "minimum": 0,
            "maximum": 9007199254740991,
        },
        "head": {
            "description": "First N lines",
            "type": "integer",
            "minimum": 0,
            "maximum": 9007199254740991,
        },
    },
    "required": ["path"],
}


class SanitizeSchemaTest(unittest.TestCase):
    def test_removes_schema_dialect(self):
        cleaned = sanitize_schema(READ_TEXT_FILE_SCHEMA)
        self.assertNotIn("$schema", cleaned)
        self.assertEqual(cleaned["type"], "object")
        self.assertEqual(cleaned["required"], ["path"])
        self.assertEqual(set(cleaned["properties"]), {"path", "tail", "head"})

    def test_preserves_constraints(self):
        cleaned = sanitize_schema(READ_TEXT_FILE_SCHEMA)
        self.assertEqual(cleaned["properties"]["tail"]["minimum"], 0)
        self.assertEqual(cleaned["properties"]["tail"]["maximum"], 9007199254740991)

    def test_nested_nullable_type(self):
        cleaned = sanitize_schema({
            "type": "object",
            "properties": {"value": {"type": ["string", "null"]}},
        })
        self.assertEqual(cleaned["properties"]["value"], {"type": "string", "nullable": True})

    def test_drops_keys_unsupported_by_gemini_schema(self):
        cleaned = sanitize_schema({
            "type": "object",
            "properties": {
                "depth": {"type": "integer", "exclusiveMinimum": 0},
                "choice": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            },
        })
        self.assertNotIn("exclusiveMinimum", cleaned["properties"]["depth"])
        self.assertNotIn("oneOf", cleaned["properties"]["choice"])

    def test_no_argument_schema_gets_object_root(self):
        cleaned = sanitize_schema({})
        self.assertEqual(cleaned, {"type": "object", "properties": {}})

    def test_input_not_mutated(self):
        original = {
            "type": "object",
            "$schema": "x",
            "properties": {"path": {"type": "string"}},
        }
        before = dict(original)
        sanitize_schema(original)
        self.assertEqual(original, before)


class DeclarationTest(unittest.TestCase):
    def setUp(self):
        self.tool = MCPToolInfo(
            name="read_text_file",
            description="Read a text file",
            input_schema=READ_TEXT_FILE_SCHEMA,
        )

    def test_declaration_name_description_and_schema(self):
        declaration = to_gemini_declaration(self.tool)
        self.assertIsInstance(declaration, types.FunctionDeclaration)
        self.assertEqual(declaration.name, "read_text_file")
        self.assertIn("[MCP filesystem]", declaration.description)
        self.assertIn("Read a text file", declaration.description)
        self.assertEqual(declaration.parameters_json_schema["required"], ["path"])
        self.assertNotIn("$schema", declaration.parameters_json_schema)

    def test_public_name_override(self):
        declaration = to_gemini_declaration(self.tool, public_name="mcp_fs_read_text_file")
        self.assertEqual(declaration.name, "mcp_fs_read_text_file")

    def test_unique_name_avoids_collision(self):
        self.assertEqual(unique_public_name("read_text_file", set()), "read_text_file")
        self.assertEqual(unique_public_name("calculate", {"calculate"}), "mcp_fs_calculate")
        self.assertEqual(
            unique_public_name("calculate", {"calculate", "mcp_fs_calculate"}),
            "mcp_fs_calculate_2",
        )


if __name__ == "__main__":
    unittest.main()
