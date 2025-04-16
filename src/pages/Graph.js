import React, { useState, useEffect } from 'react';
import { useOutletContext, useNavigate,useLocation } from 'react-router-dom';
import VoiceButton from '../components/VoiceButton';
import TextInput from '../components/TextInput';
import Papa from 'papaparse';
import "./Graph.css"

function Graph() {
  const location = useLocation();
  const {
    selectionIndex,
    setSelectionIndex,
    textSizeModifier,
    brightnessIndex,
    setBrightnessIndex,
    simplify
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
  const [allColumns, setAllColumns] = useState([]); 

  
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
        sampleValues: [],
        hasNulls: false
      };
    });
    
    // Analyze each row
    data.forEach(row => {
      Object.entries(row).forEach(([colName, value]) => {
        const col = info[colName];
        
        // Check for null values
        if (value === null || value === undefined || 
            (typeof value === 'string' && 
             ['na', 'n/a', 'null', ''].includes(value.toString().toLowerCase().trim()))) {
          col.hasNulls = true;
          return; // Skip further processing for this value
        }
        
        const parsed = parseValue(value);
        
        // Type detection - don't change from number to string just because of nulls
        if (col.type === 'unknown') {
          col.type = parsed.type;
        } else if (col.type !== parsed.type && !(col.type === 'number' && parsed.type === 'null')) {
          // Only change type if it's not a number column encountering a null
          if (col.type === 'number' && parsed.type === 'string') {
            // Try to convert string to number before changing type
            const numValue = Number(parsed.value);
            if (!isNaN(numValue)) {
              // It's a numeric string, treat as number
              parsed.type = 'number';
              parsed.value = numValue;
            } else {
              col.type = 'string';
            }
          } else {
            col.type = 'string';
          }
        }
        
        // Numeric ranges - only update if it's a valid number
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
    
    // Post-analysis: Second pass to verify number columns
    Object.entries(info).forEach(([colName, col]) => {
      // If a column has some numeric values and some nulls, keep it as number
      if (col.type === 'unknown' && col.hasNulls) {
        // Check if there are any numeric values in the column
        const hasNumbers = data.some(row => {
          const value = row[colName];
          if (value === null || value === undefined || 
              (typeof value === 'string' && 
               ['na', 'n/a', 'null', ''].includes(value.toString().toLowerCase().trim()))) {
            return false;
          }
          const numValue = Number(value);
          return !isNaN(numValue);
        });
        col.type = hasNumbers ? 'number' : 'string';
      }
    });
    
    // Convert Sets to Arrays and clean up
    Object.values(info).forEach(col => {
      col.uniqueValues = Array.from(col.uniqueValues)
        .filter(val => val !== null && val !== undefined)
        .sort();
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
  console.log(activeFilterColumns)
  console.log(filters)
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  
  // Render filter control for a specific column
  const renderColumnFilter = (colName) => {
    if (!colName || !columnInfo[colName]) return null;
    
    const info = columnInfo[colName];
    console.log(columnInfo)
    return (
      <div className="column-filter" key={colName} style={{
        border: "1px solid #ddd", 
        borderRadius: "8px", 
        padding: "10px", 
        marginBottom: "15px",
        backgroundColor: "#f9f9f9"
      }}>
        
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
          <div className="filter-controls" style={{ display: "flex", flexDirection: "row", alignItems: "center" }}>
            <span style={{ width: "107px", textAlign: "center", color: "black" }}>{colName}</span>
            <select 
              value={numInput.operation}
              onChange={(e) => handleNumericOperationChange(colName, e.target.value)}
              style={{ marginRight: "10px" }}
            >
              <option value="greaterThan">Greater than</option>
              <option value="lessThan">Less than</option>
              <option value="equals">Equal to</option>
              <option value="between">Between</option>
              <option value="notBetween">Not between</option>
            </select>
      
            <div className="input-group" style={{ display: "flex", alignItems: "center" }}>
              <input
                type="number"
                value={numInput.value1}
                onChange={(e) => handleNumericInputChange(colName, 'value1', e.target.value)}
                step="any"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') applyNumericFilter(colName);
                }}
                style={{ width: "80px", marginRight: "5px" }}
              />
      
              {(numInput.operation === 'between' || numInput.operation === 'notBetween') && (
                <>
                  <span style={{ margin: "0 5px" }}>and</span>
                  <input
                    type="number"
                    value={numInput.value2}
                    onChange={(e) => handleNumericInputChange(colName, 'value2', e.target.value)}
                    step="any"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') applyNumericFilter(colName);
                    }}
                    style={{ width: "80px", marginRight: "5px" }}
                  />
                </>
              )}
            </div>
      
            <button 
              onClick={() => applyNumericFilter(colName)}
              style={{ marginRight: "5px" }}
            >
              Apply
            </button>
      
            <button onClick={() => removeColumnFilter(colName)}>
              Remove
            </button>
      
            {filters[colName] && (
              <div style={{ marginLeft: "10px", fontStyle: "italic" }}>
                Active filter: {filters[colName].value.operation === 'between' ||
                filters[colName].value.operation === 'notBetween'
                  ? `${filters[colName].value.operation} ${filters[colName].value.value[0]} and ${filters[colName].value.value[1]}`
                  : `${filters[colName].value.operation} ${filters[colName].value.value}`}
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
          <div className="string-filter" style={{display: "flex", alignItems: "center"}}>
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
                style={{marginRight: "5px", }}
              />
              <label  htmlFor={`enable-${colName}-filter`}>Enable filter</label>
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
      <div className="data-table-container" style={{ overflowX: 'auto', maxHeight: '250px', overflowY: 'auto',marginBottom:"20px" }}>
        <table style={{color:!simplify ? "aliceblue":"black"}} className="data-table" >
          <thead style={{ position: 'sticky', top: 0, backgroundColor: '#fff', zIndex: 1 }}>
            <tr>
              {columns.map(col => (
                <th  key={col} onClick={() => handleSort(col)} style={{ cursor: 'pointer',color:"black" }}>
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
  const simplifiedStyles = {
    container: {
      
      display: 'flex',
      justifyContent:'center',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '20px',
      backgroundColor: '#fff',
      minHeight: '100vh',
      fontFamily: 'Arial, sans-serif'
    },
    heading: {
      fontSize: `${24 * textSizeModifier}px`,
      color: '#333',
      marginBottom: '15px'
    },
    fileUploader: {
      width: '100%',
      maxWidth: '800px',
      padding: '20px',
      backgroundColor: '#f8f8f8',
      borderRadius: '8px',
      boxShadow: '0 2px 5px rgba(0,0,0,0.1)',
      marginBottom: '20px',
      textAlign: 'center'
    },
    fileInput: {
      marginBottom: '15px',
      width: '100%',
      padding: '10px',
      border: '1px solid #ddd',
      borderRadius: '4px'
    },
    loadingMessage: {
      color: '#666',
      margin: '10px 0'
    },
    errorMessage: {
      color: '#d32f2f',
      fontWeight: 'bold',
      margin: '10px 0'
    },
    dataExplorer: {
      width: '100%',
      maxWidth: '1000px',
      display: 'flex',
      flexDirection: 'column',
      gap: '20px'
    },
    filtersContainer: {
      backgroundColor: '#f8f8f8',
      paddingLeft: '20px',
      paddingRight: '20px',
      borderRadius: '8px',
      boxShadow: '0 2px 5px rgba(0,0,0,0.1)'
    },
    filterSelectorRow: {
      display: 'flex',
      alignItems: 'center',
      marginBottom: '20px',
      padding: '10px',
      border: '1px dashed #aaa',
      borderRadius: '8px'
    },
    select: {
      marginRight: '10px',
      minWidth: '200px',
      padding: '8px',
      border: '1px solid #ddd',
      borderRadius: '4px'
    },
    button: {
      backgroundColor: (props) => props.disabled ? '#ccc' : '#4CAF50',
      color: 'white',
      padding: '8px 15px',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer'
    },
    activeFiltersMessage: {
      fontStyle: 'italic',
      marginBottom: '15px',
      color: '#666'
    },
    filterDivider: {
      marginTop: '20px',

    },
    dataPreview: {
      backgroundColor: '#f8f8f8',
      padding: '20px',
      borderRadius: '8px',
      boxShadow: '0 2px 5px rgba(0,0,0,0.1)',
      width: '100%'
    },
    dataPreviewHeading: {
      fontSize: `${20 * textSizeModifier}px`,
      color: '#333',
      marginBottom: '15px'
    },
    table: {
      width: '100%',
      borderCollapse: 'collapse',
      backgroundColor: '#fff',
      tableLayout: "auto",
      color: 'black',

    },
    tableHeader: {
      backgroundColor: '#f1f1f1',
      padding: '10px',
      textAlign: 'left',
      borderBottom: '2px solid #ddd'
    },
    tableCell: {
      padding: '8px',
      borderBottom: '1px solid #ddd'
    }
  };

  if (simplify) {
    return (
      <div style={simplifiedStyles.container}>
                {isLoading && csvData.length === 0 && (    
        <div style={simplifiedStyles.fileUploader}>
          <h2 style={simplifiedStyles.heading}>Upload CSV File</h2>
          <input 
            type="file" 
            accept=".csv" 
            onChange={handleFileUpload} 
            style={simplifiedStyles.fileInput}
          />
          {isLoading && file && (
            <div style={simplifiedStyles.loadingMessage}>Loading and analyzing data...</div>
          )}
          {errorMessage && (
            <div style={simplifiedStyles.errorMessage}>{errorMessage}</div>
          )}
        </div>
                )}
        {!isLoading && csvData.length > 0 && (
          <div style={simplifiedStyles.dataExplorer}>
            <div style={simplifiedStyles.filtersContainer}>
              <h2 style={simplifiedStyles.heading}>Filters</h2>
              
              <div style={simplifiedStyles.filterSelectorRow}>
                <select 
                  value=""
                  onChange={(e) => {
                    if (e.target.value) {
                      addColumnFilter(e.target.value);
                      e.target.value = ""; // Reset after selection
                    }
                  }}
                  style={simplifiedStyles.select}
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
                    const select = document.querySelector('select');
                    if (select.value) {
                      addColumnFilter(select.value);
                      select.value = ""; // Reset after selection
                    }
                  }}
                  disabled={allColumns.length === 0 || allColumns.length === activeFilterColumns.length}
                  style={{
                    backgroundColor: (allColumns.length === 0 || allColumns.length === activeFilterColumns.length) ? '#ccc' : '#4CAF50',
                    color: 'white',
                    padding: '8px 15px',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Add Filter
                </button>
              </div>
              
              <div style={{height:"120px",overflowX:"auto"}}>
                {activeFilterColumns.length === 0 ? (
                  <div style={simplifiedStyles.activeFiltersMessage}>
                    No column filters added. Select columns to create filters.
                  </div>
                ) : (
                  activeFilterColumns.map(colName => renderColumnFilter(colName))
                )}
              </div>
              
              {activeFilterColumns.length > 0 && (
                <div style={simplifiedStyles.filterDivider}>
                  {renderActiveFilters()}
                </div>
              )}
            </div>
            
            <div style={simplifiedStyles.dataPreview}>
              <h2 style={simplifiedStyles.dataPreviewHeading}>
                Filtered Data ({filteredData.length} rows)
              </h2>
              {renderDataTable(true)} {/* Pass a flag to use simplified styling */}
            </div>
          </div>
        )}
        
        <VoiceButton
        setSelectionIndex={setSelectionIndex}
        selectionIndex={selectionIndex}
        page={location.pathname}
        columnInfo={columnInfo}
        setActiveFilterColumns={setActiveFilterColumns}
        setFilters={setFilters}
      />
      </div>
    );
  }

  return (
    <div className="body" style={{ filter: `brightness(${1 * brightnessIndex})` }}>
      <div className="graph">
        {isLoading && csvData.length === 0 && (        <div className="csv-uploader">
          <h2 style={{ fontSize: `${24 * textSizeModifier}px` }}>Upload CSV File</h2>
          <input 
            style={{ marginBottom: "15px", fontSize: `${14 * textSizeModifier}px` }} 
            type="file" 
            accept=".csv" 
            onChange={handleFileUpload} 
          />
          {isLoading && file && (
            <div style={{ fontSize: `${16 * textSizeModifier}px` }}>Loading and analyzing data...</div>
          )}
          {errorMessage && (
            <div className="error" style={{ fontSize: `${16 * textSizeModifier}px` }}>{errorMessage}</div>
          )}
        </div>)}

        
        {!isLoading && csvData.length > 0 && (
          <div className="data-explorer">
            <div className="filters" style={{width:"750px"}}>
              <h2 style={{ fontSize: `${24 * textSizeModifier}px` }}>Filters</h2>
              
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
                  style={{ marginRight: "10px", minWidth: "200px", fontSize: `${14 * textSizeModifier}px` }}
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
                    borderRadius: "4px",
                    fontSize: `${14 * textSizeModifier}px`
                  }}
                >
                  Add Filter
                </button>
              </div>
              
              <div className="active-column-filters">
                {activeFilterColumns.length === 0 ? (
                  <div style={{ fontStyle: "italic", marginBottom: "15px", fontSize: `${14 * textSizeModifier}px` }}>
                    No column filters added. Select columns to create filters.
                  </div>
                ) : (
                  activeFilterColumns.map(colName => renderColumnFilter(colName))
                )}
              </div>
              
              {activeFilterColumns.length > 0 && (
                <div style={{ marginTop: "20px", borderTop: "1px solid #ddd", paddingTop: "10px" }}>
                  {renderActiveFilters()}
                </div>
              )}
            </div>
            
            <div className="data-preview" style={{width:"750px"}}>
              <h2 style={{ fontSize: `${24 * textSizeModifier}px` }}>
                Filtered Data ({filteredData.length} rows)
              </h2>
              {renderDataTable(false)} {/* Pass flag to use original styling */}
            </div>
          </div>
        )}
      </div>
      <VoiceButton
        setSelectionIndex={setSelectionIndex}
        selectionIndex={selectionIndex}
        page={location.pathname}
        columnInfo={columnInfo}
        setActiveFilterColumns={setActiveFilterColumns}
        setFilters={setFilters}
      />
    </div>
  );
}

export default Graph;