import pytest
from fastapi.testclient import TestClient

from app import app, command, commandNavigation

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_commands():
    """
    Before each test, reset and populate the `command` and `commandNavigation` lists
    with minimal dummy data to make `send_message` and `send_message_navigation` work predictably.
    """
    command.clear()
    command.append({"id": "cmd1", "name": "hello world"})
    commandNavigation.clear()
    commandNavigation.append({"id": "nav1", "name": "navigate home"})
    yield

    command.clear()
    commandNavigation.clear()


@pytest.mark.parametrize(
    "endpoint,payload,expected_key",
    [
        ("/send-message", {"message": "hello world"}, "text"),
        ("/send-message-navigation", {"message": "navigate home"}, "text"),
    ],
)
def test_similarity_endpoints(endpoint, payload, expected_key):
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first_item = data[0]
    assert expected_key in first_item
    # Basic assertion that the returned text matches our dummy command
    assert first_item[expected_key] in [
        item["name"]
        for item in (command if endpoint == "/send-message" else commandNavigation)
    ]


def test_predict_message():

    payload = {"message": "test prediction"}
    response = client.post("/predict-message", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, str)


def test_filter_endpoint():

    payload = {
        "message": "set volume to 5 and mode to quiet",
    }
    response = client.post("/send-message", json=payload)
    assert response.status_code == 200
    result = response.json()
    print(result)


def test_get_issue():

    payload = {
        "keys": [
            {
                "time": "2025-05-19T12:00:00Z",
                "positionX": 10.0,
                "positionY": 20.0,
                "eyeX": 300.0,
                "eyeY": 400.0,
                "hoverType": "hover",
                "isMouseDown": False,
                "scrollDirection": "up",
            }
            for _ in range(250)
        ]
    }
    response = client.post("/get-issue", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert isinstance(result, int)


def test_filter_sentence_endpoint():
    # Test the /filter_sentence endpoint with multiple conditions
    payload = {
        "message": "I want the species to be adelie, the culmen length to be between 20 and 21, and flipper length to be more than 190",
        "keys": {
            "species": {
                "type": "string",
                "uniqueValues": ["Adelie", "Chinstrap", "Gentoo"],
            },
            "island": {
                "type": "string",
                "uniqueValues": ["Biscoe", "Dream", "Torgersen"],
            },
            "culmen_length_mm": {"type": "number", "uniqueValues": [32.1, 58]},
            "culmen_depth_mm": {"type": "number", "uniqueValues": [13.1, 21.5]},
            "flipper_length_mm": {"type": "number", "uniqueValues": [172, 231]},
            "body_mass_g": {"type": "number", "uniqueValues": [2700, 6300]},
            "sex": {"type": "string", "uniqueValues": [".", "FEMALE", "MALE"]},
        },
    }
    response = client.post("/filter_sentence", json=payload)
    assert response.status_code == 200
    result = response.json()
    # Should return a dict with 'sentence' and 'target_string'
    assert isinstance(result, dict)
    assert "sentence" in result
    assert "target_string" in result
    # Validate that species adelie is captured
    targets = result["target_string"]
    assert "species" in targets
    assert targets["species"] == [1]  # Adelie is the first value
    # Validate numerical conditions
    assert "culmen_length_mm" in targets
    # Should capture any numbers between 20 and 21 if within unique range
    assert all(20 <= val <= 21 for val in targets["culmen_length_mm"])
    assert "flipper_length_mm" in targets
    assert all(val > 190 for val in targets["flipper_length_mm"])


def test_filter_sentence_endpoint():
    # Test the /filter_sentence endpoint with multiple conditions
    payload = {
        "message": "I want the species to be adelie, the culmen length to be between 20 and 21, and flipper length to be more than 190",
        "keys": {
            "species": {
                "type": "string",
                "uniqueValues": ["Adelie", "Chinstrap", "Gentoo"],
            },
            "island": {
                "type": "string",
                "uniqueValues": ["Biscoe", "Dream", "Torgersen"],
            },
            "culmen_length_mm": {"type": "number", "uniqueValues": [32.1, 58]},
            "culmen_depth_mm": {"type": "number", "uniqueValues": [13.1, 21.5]},
            "flipper_length_mm": {"type": "number", "uniqueValues": [172, 231]},
            "body_mass_g": {"type": "number", "uniqueValues": [2700, 6300]},
            "sex": {"type": "string", "uniqueValues": [".", "FEMALE", "MALE"]},
        },
    }
    response = client.post("/filter_sentence", json=payload)
    assert response.status_code == 200
    result = response.json()
    # Should return a dict with 'sentence' and 'target_string'
    assert isinstance(result, dict)
    assert "sentence" in result
    assert "target_string" in result
    # Validate that species adelie is captured
    targets = result["target_string"]
    assert "species" in targets
    # Validate numerical conditions
    assert "culmen_length_mm" in targets
    # Should capture any numbers between 20 and 21 if within unique range
    assert all(20 <= val <= 21 for val in targets["culmen_length_mm"])
    assert "flipper_length_mm" in targets
    assert all(val > 190 for val in targets["flipper_length_mm"])
