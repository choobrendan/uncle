import React, { useState, useEffect } from 'react';
import { useOutletContext, useNavigate } from 'react-router-dom';
import VoiceButton from '../components/VoiceButton';
import TextInput from '../components/TextInput';
import Papa from 'papaparse';
import "./Graph.css"

function Graph() {
  const {
    selectionIndex,
    setSelectionIndex,
    textSizeModifier,
    brightnessIndex,
    setBrightnessIndex
  } = useOutletContext();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [csvData, setCsvData] = useState([]);
  const [columnInfo, setColumnInfo] = useState({});
  const [filters, setFilters] = useState({});
  const [filteredData, setFilteredData] = useState([]);
  const [file, setFile] = useState(null);
  const [numericFilterInputs, setNumericFilterInputs] = useState({});
  const [activeFilterColumns, setActiveFilterColumns] = useState([]); // Track active filter columns
  const [allColumns, setAllColumns] = useState([]); // Track all available columns
  
  // Handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setFile(file);
    setIsLoading(true);
    
    Papa.parse(file, {
      header: true,
      complete: (results) => {
        setCsvData(results.data);
        analyzeData(results.data);
        setIsLoading(false);
      },
      error: (error) => {
        setErrorMessage("Error parsing CSV: " + error.message);
        setIsLoading(false);
      }
    });
  };

  // Analyze data structure
  const analyzeData = (data) => {
    if (data.length === 0) return;
    const firstRow = data[0];
    const info = {};
    
    // Initialize column structure
    Object.keys(firstRow).forEach(key => {
      info[key] = {
        type: 'unknown',
        min: Infinity,
        max: -Infinity,
        uniqueValues: new Set(),
        sampleValues: []
      };
    });
    
    // Analyze each row
    data.forEach(row => {
      Object.entries(row).forEach(([colName, value]) => {
        const col = info[colName];
        const parsed = parseValue(value);
        
        // Type detection
        if (col.type === 'unknown') col.type = parsed.type;
        else if (col.type !== parsed.type) col.type = 'string';
        
        // Numeric ranges
        if (parsed.type === 'number') {
          col.min = Math.min(col.min, parsed.value);
          col.max = Math.max(col.max, parsed.value);
        }
        
        // Unique values (limit to 100 for performance)
        if (col.uniqueValues.size < 100) {
          col.uniqueValues.add(parsed.value);
        }
      });
    });
    
    // Convert Sets to Arrays and clean up
    Object.values(info).forEach(col => {
      col.uniqueValues = Array.from(col.uniqueValues).sort();
      if (col.min === Infinity) delete col.min;
      if (col.max === -Infinity) delete col.max;
    });
    
    setColumnInfo(info);
    setFilteredData(data); // Initially show all data
    
    // Initialize numeric filter inputs
    const initialNumericInputs = {};
    Object.entries(info).forEach(([colName, colInfo]) => {
      if (colInfo.type === 'number') {
        initialNumericInputs[colName] = {
          operation: 'between',
          value1: colInfo.min || 0,
          value2: colInfo.max || 0
        };
      }
    });
    setNumericFilterInputs(initialNumericInputs);
    
    // Set available columns - now we store them in allColumns
    const columns = Object.keys(info);
    setAllColumns(columns);
    setActiveFilterColumns([]); // Reset active filter columns
    setFilters({}); // Reset filters
  };

  const parseValue = (value) => {
    if (value === null || value === undefined) return { type: 'string', value: '' };
    
    // Try parsing as number
    if (!isNaN(value) && value.toString().trim() !== '') {
      const num = parseFloat(value);
      return { type: 'number', value: num };
    }
    
    // Try parsing as date
    const date = new Date(value);
    if (!isNaN(date)) return { type: 'date', value: date };
    
    // Default to string
    return { type: 'string', value: value.toString().trim() };
  };

  // Add a column filter
  const addColumnFilter = (colName) => {
    
    setActiveFilterColumns(prev => [...prev, colName]);
    // No longer removing from available columns
  };

  // Remove a column filter
  const removeColumnFilter = (colName) => {
    // Remove from active filters
    setActiveFilterColumns(prev => prev.filter(col => col !== colName));
    
    // Remove any filter for this column
    handleFilterChange(colName, null, null);
  };

  const handleFilterChange = (colName, type, value) => {
    setFilters(prev => {
      if (value === null) {
        // Remove filter
        const newFilters = { ...prev };
        delete newFilters[colName];
        return newFilters;
      } else {
        // Add or update filter
        return {
          ...prev,
          [colName]: { type, value }
        };
      }
    });
  };

  const handleNumericOperationChange = (colName, operation) => {
    setNumericFilterInputs(prev => ({
      ...prev,
      [colName]: {
        ...prev[colName],
        operation
      }
    }));
  };

  const handleNumericInputChange = (colName, inputName, value) => {
    setNumericFilterInputs(prev => ({
      ...prev,
      [colName]: {
        ...prev[colName],
        [inputName]: value
      }
    }));
  };

  const applyNumericFilter = (colName) => {
    const input = numericFilterInputs[colName];
    const value1 = parseFloat(input.value1);
    const value2 = parseFloat(input.value2);
    
    let filterValue;
    switch (input.operation) {
      case 'greaterThan':
        filterValue = { operation: 'greaterThan', value: value1 };
        break;
      case 'lessThan':
        filterValue = { operation: 'lessThan', value: value1 };
        break;
      case 'equals':
        filterValue = { operation: 'equals', value: value1 };
        break;
      case 'between':
        filterValue = { operation: 'between', value: [value1, value2] };
        break;
      case 'notBetween':
        filterValue = { operation: 'notBetween', value: [value1, value2] };
        break;
      default:
        return;
    }
    
    handleFilterChange(colName, 'number', filterValue);
  };

  // Apply filters when they change
  useEffect(() => {
    if (csvData.length === 0) return;
    const filtered = csvData.filter(row => {
      return Object.entries(filters).every(([colName, filter]) => {
        if (!filter) return true; // No filter set for this column
        
        const parsed = parseValue(row[colName]);
        const value = parsed.value;
        
        switch (filter.type) {
          case 'number':
            if (filter.value.operation === 'greaterThan') {
              return value > filter.value.value;
            } else if (filter.value.operation === 'lessThan') {
              return value < filter.value.value;
            } else if (filter.value.operation === 'equals') {
              return value === filter.value.value;
            } else if (filter.value.operation === 'between') {
              return value >= filter.value.value[0] && value <= filter.value.value[1];
            } else if (filter.value.operation === 'notBetween') {
              return value < filter.value.value[0] || value > filter.value.value[1];
            }
            return true;
            
          case 'date':
            const dateValue = new Date(value);
            return dateValue >= new Date(filter.value[0]) && 
                   dateValue <= new Date(filter.value[1]);
            
          case 'string':
            if (!filter.value) return true;
            return filter.value.includes(value.toString());
            
          default:
            return true;
        }
      });
    });
    setFilteredData(filtered);
  }, [filters, csvData]);

  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  
  // Render filter control for a specific column
  const renderColumnFilter = (colName) => {
    if (!colName || !columnInfo[colName]) return null;
    
    const info = columnInfo[colName];
    
    return (
      <div className="column-filter" key={colName} style={{
        border: "1px solid #ddd", 
        borderRadius: "8px", 
        padding: "10px", 
        marginBottom: "15px",
        backgroundColor: "#f9f9f9"
      }}>
        <div style={{display: "flex", justifyContent: "space-between", marginBottom: "10px"}}>
          <h3 style={{margin: 0}}>{colName} ({info.type})</h3>
          <button 
            onClick={() => removeColumnFilter(colName)}
            style={{background: "#ff6b6b", color: "white", border: "none", borderRadius: "4px", padding: "4px 8px"}}
          >
            Remove
          </button>
        </div>
        
        {renderFilterControlsByType(colName, info)}
      </div>
    );
  };
  
  // Render filter controls based on column type
  const renderFilterControlsByType = (colName, info) => {
    switch (info.type) {
      case 'number':
        const numInput = numericFilterInputs[colName] || {
          operation: 'between',
          value1: info.min || 0,
          value2: info.max || 0
        };
        
        return (
          <div className="filter-controls" style={{display:"flex", flexDirection:"row", alignItems:"center", marginTop: "10px"}}>
            <select 
              value={numInput.operation}
              onChange={(e) => handleNumericOperationChange(colName, e.target.value)}
              style={{marginRight: "10px"}}
            >
              <option value="greaterThan">Greater than</option>
              <option value="lessThan">Less than</option>
              <option value="equals">Equal to</option>
              <option value="between">Between</option>
              <option value="notBetween">Not between</option>
            </select>
            
            <div className="input-group" style={{display: "flex", alignItems: "center"}}>
              <input
                type="number"
                value={numInput.value1}
                onChange={(e) => handleNumericInputChange(colName, 'value1', e.target.value)}
                min={info.min}
                max={info.max}
                step="any"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') applyNumericFilter(colName);
                }}
                style={{width: "80px", marginRight: "5px"}}
              />
              
              {(numInput.operation === 'between' || numInput.operation === 'notBetween') && (
                <>
                  <span style={{margin: "0 5px"}}>and</span>
                  <input
                    type="number"
                    value={numInput.value2}
                    onChange={(e) => handleNumericInputChange(colName, 'value2', e.target.value)}
                    min={info.min}
                    max={info.max}
                    step="any"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') applyNumericFilter(colName);
                    }}
                    style={{width: "80px", marginRight: "5px"}}
                  />
                </>
              )}
            </div>
            
            <button 
              onClick={() => applyNumericFilter(colName)}
              style={{marginRight: "5px"}}
            >
              Apply
            </button>

            <button 
              onClick={() => handleFilterChange(colName, 'number', null)}
            >
              Clear
            </button>

            {filters[colName] && (
              <div style={{marginLeft: "10px", fontStyle: "italic"}}>
                Active filter: {filters[colName].value.operation === 'between' || filters[colName].value.operation === 'notBetween' ? 
                  `${filters[colName].value.operation} ${filters[colName].value.value[0]} and ${filters[colName].value.value[1]}` : 
                  `${filters[colName].value.operation} ${filters[colName].value.value}`
                }
              </div>
            )}
          </div>
        );
        
      case 'date':
        return (
          <div className="date-filter" style={{display: "flex", alignItems: "center", marginTop: "10px"}}>
            <label style={{marginRight: "5px"}}>From:</label>
            <input
              type="date"
              min={info.min?.toISOString().split('T')[0]}
              onChange={(e) => handleFilterChange(
                colName,
                'date',
                [e.target.value, filters[colName]?.value?.[1] || info.max?.toISOString().split('T')[0]]
              )}
              style={{marginRight: "10px"}}
            />
            <label style={{marginRight: "5px"}}>To:</label>
            <input
              type="date"
              max={info.max?.toISOString().split('T')[0]}
              onChange={(e) => handleFilterChange(
                colName,
                'date',
                [filters[colName]?.value?.[0] || info.min?.toISOString().split('T')[0], e.target.value]
              )}
              style={{marginRight: "10px"}}
            />
            {filters[colName] && (
              <>
                <button 
                  onClick={() => handleFilterChange(colName, 'date', null)}
                  style={{marginRight: "10px"}}
                >
                  Clear
                </button>
                <div style={{fontStyle: "italic"}}>
                  Active filter: From {filters[colName].value[0]} to {filters[colName].value[1]}
                </div>
              </>
            )}
          </div>
        );
        
      default: // string type
        return (
          <div className="string-filter" style={{display: "flex", alignItems: "center", marginTop: "10px"}}>
            <div className="filter-header" style={{marginRight: "10px"}}>
              <input
                type="checkbox"
                id={`enable-${colName}-filter`}
                checked={!!filters[colName]}
                onChange={(e) => {
                  if (e.target.checked) {
                    const firstValue = info.uniqueValues[0];
                    handleFilterChange(colName, 'string', [firstValue]);
                  } else {
                    handleFilterChange(colName, 'string', null);
                  }
                }}
                style={{marginRight: "5px"}}
              />
              <label htmlFor={`enable-${colName}-filter`}>Enable filter</label>
            </div>
            
            <select
              disabled={!filters[colName]}
              value={filters[colName]?.value?.[0] || ''}
              onChange={(e) => {
                const selected = e.target.value;
                handleFilterChange(colName, 'string', [selected]);
              }}
              style={{marginRight: "10px", minWidth: "120px"}}
            >
              {!filters[colName] && <option value="">Select a value</option>}
              {info.uniqueValues.map((value, i) => (
                <option key={i} value={value}>{value || '(empty)'}</option>
              ))}
            </select>
            
            {filters[colName] && (
              <div style={{fontStyle: "italic"}}>
                Active filter: {filters[colName].value[0] || '(empty)'}
              </div>
            )}
          </div>
        );
    }
  };

  // Function to render active filters summary
  const renderActiveFilters = () => {
    const activeFilters = Object.entries(filters).filter(([colName, filter]) => filter !== null);
    
    if (activeFilters.length === 0) return <div>No active filters</div>;
    
    return (
      <div className="active-filters">
      </div>
    );
  };

  const renderDataTable = () => {
    // Sort the data based on the current sortConfig
    const sortedData = [...filteredData].sort((a, b) => {
      if (sortConfig.key === null) return 0;
  
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];
  
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortConfig.direction === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }
  
      return sortConfig.direction === 'asc'
        ? aValue - bValue
        : bValue - aValue;
    });
  
    const columns = Object.keys(filteredData[0] || {});
  
    const handleSort = (column) => {
      let direction = 'asc';
      if (sortConfig.key === column && sortConfig.direction === 'asc') {
        direction = 'desc';
      }
      setSortConfig({ key: column, direction });
    };
  
    if (filteredData.length === 0) return <div>No data to display</div>;
  
    return (
      <div className="data-table-container" style={{ overflowX: 'auto', maxHeight: '400px', overflowY: 'auto' }}>
        <table className="data-table">
          <thead style={{ position: 'sticky', top: 0, backgroundColor: '#fff', zIndex: 1 }}>
            <tr>
              {columns.map(col => (
                <th key={col} onClick={() => handleSort(col)} style={{ cursor: 'pointer' }}>
                  {col} {sortConfig.key === col ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedData.map((row, i) => (
              <tr key={i}>
                {columns.map(col => (
                  <td key={`${i}-${col}`}>{row[col]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="body">
      <div className="graph">
        <div className="csv-uploader">
          <h2>Upload CSV File</h2>
          <input style={{marginBottom: "15px"}} type="file" accept=".csv" onChange={handleFileUpload} />
          {isLoading && file && <div>Loading and analyzing data...</div>}
          {errorMessage && <div className="error">{errorMessage}</div>}
        </div>
        
        {!isLoading && csvData.length > 0 && (
          <div className="data-explorer">
            <div className="filters">
              <h2>Filters</h2>
              
              {/* Column selector dropdown and add button */}
              <div className="add-column-filter" style={{
                display: "flex", 
                alignItems: "center", 
                marginBottom: "20px",
                border: "1px dashed #aaa",
                padding: "10px",
                borderRadius: "8px"
              }}>
                <select 
                  value=""
                  onChange={(e) => {
                    if (e.target.value) {
                      addColumnFilter(e.target.value);
                      e.target.value = ""; // Reset after selection
                    }
                  }}
                  style={{marginRight: "10px", minWidth: "200px"}}
                >
                  <option value="">-- Select column to add filter --</option>
                  {allColumns.map(colName => (
    <option key={colName} value={colName}>
        {colName} ({columnInfo[colName]?.type})
    </option>
))}
                </select>
                
                <button 
                  onClick={() => {
                    const select = document.querySelector('.add-column-filter select');
                    if (select.value) {
                      addColumnFilter(select.value);
                      select.value = ""; // Reset after selection
                    }
                  }}
                  disabled={allColumns.length === 0 || allColumns.length === activeFilterColumns.length}
                  style={{
                    background: (allColumns.length === 0 || allColumns.length === activeFilterColumns.length) ? "#ccc" : "#4CAF50",
                    color: "white",
                    padding: "5px 10px",
                    border: "none",
                    borderRadius: "4px"
                  }}
                >
                  Add Filter
                </button>
              </div>
              <div className="active-column-filters">
                {activeFilterColumns.length === 0 ? (
                  <div style={{fontStyle: "italic", marginBottom: "15px"}}>No column filters added. Select columns to create filters.</div>
                ) : (
                  activeFilterColumns.map(colName => renderColumnFilter(colName))
                )}
              </div>
              
              {activeFilterColumns.length > 0 && (
                <div style={{marginTop: "20px", borderTop: "1px solid #ddd", paddingTop: "10px"}}>
                  {renderActiveFilters()}
                </div>
              )}
            </div>
            
            <div className="data-preview">
              <h2>Filtered Data ({filteredData.length} rows)</h2>
              {renderDataTable()}
            </div>
          </div>
        )}
      </div>
      <VoiceButton
        setSelectionIndex={setSelectionIndex}
        selectionIndex={selectionIndex}
      />
    </div>
  );
}

export default Graph;