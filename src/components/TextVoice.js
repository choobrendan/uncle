import React, { useState, useEffect } from "react";
import TextInput from "./TextInput";
import "./TextBox.css";

const TextVoice = ({
  suggestions,
  setSelectionIndex,
  selectionIndex,
  text,
  onInput,
  page,
  columnInfo,
  setActiveFilterColumns,
  setFilters,
  simplify,
  font,
}) => {
  const [inputValue, setInputValue] = useState(text || "");
  const [filteredSuggestions, setFilteredSuggestions] = useState([]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (filteredSuggestions.length === 0) return;
      if (e.keyCode === 13 && selectionIndex >= 0) {
        selectSuggestion(filteredSuggestions[selectionIndex]);
        e.preventDefault();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [filteredSuggestions, selectionIndex]);

  const sendText = (value) => {
    console.log(value);
    fetch("http://localhost:8000/send-message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: value }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        const suggestionsList = data.map((item) => ({
          id: item.id, // Extract the 'id' from each item
          name: item.text, // Extract the 'name' from each item
        }));
        setFilteredSuggestions(suggestionsList);
        console.log(suggestionsList, "Suggestions updated");
      })
      .catch((error) => console.error("Error sending message:", error));
  };
  function wordToIndex(word) {
    const mapping = {
      one: 0,
      two: 1,
      three: 2,
      four: 3,
      five: 4,
      six: 5,
      seven: 6,
      eight: 7,
      nine: 8,
      ten: 9,
    };
    return mapping[word.toLowerCase()] ?? -1;
  }

  /**
   * Given a sentence template and an array of mapping objects (targetString),
   * replace all placeholder tags in the template with the corresponding value.
   * Each mapping object is expected to contain a single key-value pair.
   */
  function replaceTemplateTags(template, targetString) {
    let sentence = template;
    targetString.forEach((mapping) => {
      for (const tag in mapping) {
        // Use a global replace for all occurrences of the tag within angle brackets.
        const regex = new RegExp(`<${tag}>`, "g");
        sentence = sentence.replace(regex, mapping[tag]);
      }
    });
    return sentence;
  }

  // --- Main Functions ---

  /**
   * Extracts target keys from the sentence.
   * A <target word> token is used to denote a key.
   * The word (e.g., "three", "five") is converted to a 0-based index into the keys order.
   * Returns an array of keys referenced in the sentence.
   */
  function generateKeyValueArray(sentence, simplified) {
    const tokens = sentence.split(/\s+/);
    const result = [];
    // Convert keys to array (keeping order)
    const allKeys = Object.keys(simplified);

    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i].startsWith("<target")) {
        // Expect the number word to be the next token (possibly with the trailing '>')
        if (i + 1 < tokens.length) {
          const numWord = tokens[i + 1].replace(">", "");
          const index = wordToIndex(numWord);
          if (index >= 0 && index < allKeys.length) {
            result.push(allKeys[index]);
          }
        }
      }
    }
    return result;
  }

  /**
   * Parses the sentence for filter instructions, extracting both target keys and operations.
   * The function maps tokens such as "<target three>" and operations like "<between>" or "<lower>".
   * It then extracts numeric values in order and assigns them as filter values.
   *
   * Returns an object where each target key maps to a filter object that specifies:
   *   - type: (copied from simplified)
   *   - operation: the operation to perform (e.g., "between", "lessThan")
   *   - value: one or more numeric values
   */
  function generateKeyFilter(sentence, simplified) {
    const filters = {};

    // Regular expression to extract tokens enclosed in angle brackets, e.g. "target three", "between", etc.
    const tagPattern = /<([^>]+)>/g;
    const tags = [];
    let match;
    while ((match = tagPattern.exec(sentence)) !== null) {
      tags.push(match[1]);
    }

    // Extract all number sequences (as strings) using a regular expression.
    // This supports integers and decimals.
    const numberPattern = /\b\d+(?:\.\d+)?\b/g;
    const numbers = [];
    while ((match = numberPattern.exec(sentence)) !== null) {
      numbers.push(match[0]);
    }

    // Mapping operation tokens to desired operation strings.
    const opMapping = {
      between: "between",
      lower: "lessThan",
      higher: "greaterThan",
      equals: "equals",
    };

    let numIdx = 0; // pointer for numbers array
    let currentKey = null;
    const allKeys = Object.keys(simplified);

    tags.forEach((tag) => {
      const parts = tag.split(/\s+/);
      if (parts.length === 0) return;

      const tokenType = parts[0].toLowerCase();

      if (tokenType === "target") {
        // Expect format: target <numberWord>
        if (parts.length > 1) {
          const keyIndex = wordToIndex(parts[1]);
          if (keyIndex >= 0 && keyIndex < allKeys.length) {
            currentKey = allKeys[keyIndex];
            // Initialize the filter object for the key using its type from simplified
            filters[currentKey] = { type: simplified[currentKey].type };
          } else {
            currentKey = null;
          }
        } else {
          currentKey = null;
        }
      } else if (opMapping.hasOwnProperty(tokenType)) {
        const op = opMapping[tokenType];
        if (currentKey) {
          filters[currentKey]["operation"] = op;
          if (op === "between") {
            // For "between", expect two numbers.
            if (numIdx + 1 < numbers.length) {
              const value1 = parseFloat(numbers[numIdx]);
              const value2 = parseFloat(numbers[numIdx + 1]);
              filters[currentKey]["value"] = [value1, value2];
              numIdx += 2;
            } else {
              filters[currentKey]["value"] = [];
            }
          } else {
            // For single-number operations such as "lessThan", "greaterThan", or "equals".
            if (numIdx < numbers.length) {
              filters[currentKey]["value"] = {
                operation: op,
                value: parseFloat(numbers[numIdx]),
              };
              numIdx += 1;
            } else {
              filters[currentKey]["value"] = null;
            }
          }
        }
      }
    });

    return filters;
  }

  let timeoutId; // this needs to be in a scope that persists between calls

  const sendPredict = (result) => {
    // Clear any existing timeout (reset the debounce)
    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    // Set a new timeout that will fire after 1 second of inactivity
    timeoutId = setTimeout(() => {
      const simplified = {};

      // Create the simplified object for keys
      for (const [key, value] of Object.entries(columnInfo)) {
        simplified[key] = {
          type: value.type,
          uniqueValues: value.uniqueValues,
        };
      }
      //

      console.log({ keys: simplified, message: result });

      // Make the POST request
      fetch("http://localhost:8000/filter_sentence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keys: simplified, message: result }),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log(data);
          const finalSentence = replaceTemplateTags(
            data.sentence,
            data.target_string
          );
          console.log("Final Sentence:", finalSentence);

          // Generate an ordered array of target keys based on sentence tokens (activeFilterColumns)
          setActiveFilterColumns(
            generateKeyValueArray(finalSentence, simplified)
          );

          // Generate the key filter object from the final sentence (filteredData)
          setFilters(generateKeyFilter(finalSentence, simplified));
          console.log(generateKeyFilter(finalSentence, simplified));
        })
        .catch((error) => console.error("Error sending message:", error));
    }, 1000);
  };

  const selectSuggestion = (suggestion) => {
    setInputValue(suggestion);
    if (onInput) {
      onInput(suggestion);
    }
  };

  // Update input value when text prop or voice result changes
  useEffect(() => {
    if (text) {
      setInputValue(text);
    }
  }, [text]);

  // Trigger the sendText function whenever inputValue changes
  useEffect(() => {
    if (inputValue.trim()) {
      sendText(inputValue);
      console.log(page);
      if (page === "/graph") {
        sendPredict(inputValue);
      }
    }
  }, [inputValue]);

  return (
    <div className="neon-container">
      <TextInput
        simplify={simplify}
        value={inputValue}
        font={font}
        onChange={(e) => {
          setInputValue(e.target.value);
        }}
        placeholder="Type a prompt..."
      />

      {filteredSuggestions.length > 0 && (
        <div
          className="suggestion-buttons"
          style={{ display: "flex", flexDirection: "column" }}
        >
          {filteredSuggestions.slice(0, 4).map((suggestion, index) => (
            <button
              style={{
                fontFamily: font,
                fontSize: "20px",
                margin: "5px",
              }}
              key={suggestion.id}
              className={`suggestion-button ${
                selectionIndex === index ? "active" : ""
              }`}
              onClick={() => {
                selectSuggestion(suggestion.name);
                setSelectionIndex(suggestion.id);
              }}
            >
              {suggestion.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default TextVoice;
