"use client"
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { AlertCircle, ChevronLeft, ChevronRight, Save } from 'lucide-react';

interface ResponsePair {
  prompt: string;
  response1: string;
  response2: string;
  selectedIndex?: number;
  note?: string;
  timestamp?: string;
}

export const ComparisonInterface = () => {
  const [data, setData] = useState<ResponsePair[]>([]);
  const [currentPairIndex, setCurrentPairIndex] = useState(0);
  const [evaluations, setEvaluations] = useState<Map<number, ResponsePair>>(new Map());
  const [selectedResponse, setSelectedResponse] = useState<number | null>(null);
  const [note, setNote] = useState('');

  const parseCSVLine = (text: string): string[] => {
    const result: string[] = [];
    let currentField = '';
    let inQuotes = false;
    let i = 0;
    
    while (i < text.length) {
      const char = text[i];
      
      if (char === '"') {
        if (i + 1 < text.length && text[i + 1] === '"') {
          currentField += '"';
          i += 2;
        } else {
          inQuotes = !inQuotes;
          i++;
        }
      } else if (char === ',' && !inQuotes) {
        result.push(currentField);
        currentField = '';
        i++;
      } else {
        if (inQuotes || (char !== '\n' && char !== '\r')) {
          currentField += char;
        }
        i++;
      }
    }
    
    result.push(currentField);
    
    return result.map(field => {
      field = field.trim();
      if (field.startsWith('"') && field.endsWith('"')) {
        field = field.slice(1, -1);
      }
      return field.replace(/""/g, '"');
    });
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      
      let lines: string[] = [];
      let currentLine = '';
      let inQuotes = false;
      
      for (let i = 0; i < text.length; i++) {
        const char = text[i];
        
        if (char === '"') {
          inQuotes = !inQuotes;
          currentLine += char;
        } else if ((char === '\n' || char === '\r') && !inQuotes) {
          if (currentLine.trim()) {
            lines.push(currentLine);
          }
          currentLine = '';
        } else if (char !== '\r' || inQuotes) {
          currentLine += char;
        }
      }
      
      if (currentLine.trim()) {
        lines.push(currentLine);
      }

      const processedData: ResponsePair[] = [];
      const startIndex = lines[0].toLowerCase().includes('prompt') ? 1 : 0;
      
      for (let i = startIndex; i < lines.length; i++) {
        const line = lines[i];
        
        let fields: string[];
        if (line.includes('\t')) {
          fields = line.split('\t').map(field => {
            field = field.trim();
            return field.replace(/^"|"$/g, '');
          });
        } else {
          fields = parseCSVLine(line);
        }

        if (fields[0]?.trim()) {
          processedData.push({
            prompt: fields[0] || '',
            response1: fields[1] || '',
            response2: fields[2] || ''
          });
        }
      }

      setData(processedData);
      setCurrentPairIndex(0);
      setSelectedResponse(null);
      setNote('');
      setEvaluations(new Map());
    };
    reader.readAsText(file);
  };

  const formatForCSV = (value: string): string => {
    if (!value) return '""';
    const needsQuoting = value.includes(',') || value.includes('"') || value.includes('\n') || value.includes('\r');
    if (!needsQuoting) return value;
    return `"${value.replace(/"/g, '""')}"`;
  };

  const saveSelection = () => {
    if (selectedResponse === null || !data[currentPairIndex]) return;

    const currentEvaluation = {
      ...data[currentPairIndex],
      selectedIndex: selectedResponse,
      note,
      timestamp: new Date().toISOString()
    };

    const newEvaluations = new Map(evaluations);
    newEvaluations.set(currentPairIndex, currentEvaluation);
    setEvaluations(newEvaluations);

    // Auto-save to CSV after each selection
    const sortedEvaluations = Array.from(newEvaluations.entries())
      .sort(([a], [b]) => a - b)
      .map(([_, evaluation]) => evaluation);

    const csvContent = [
      ['prompt', 'response1', 'response2', 'selected_index', 'note', 'timestamp'].join(','),
      ...sortedEvaluations.map(evaluation => [
        formatForCSV(evaluation.prompt),
        formatForCSV(evaluation.response1),
        formatForCSV(evaluation.response2),
        evaluation.selectedIndex?.toString() || '',
        formatForCSV(evaluation.note || ''),
        formatForCSV(evaluation.timestamp || '')
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'response_evaluations.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const exportToCSV = () => {
    // This function now just triggers saveSelection which handles the CSV export
    saveSelection();
  };

  const goToPreviousPair = () => {
    if (currentPairIndex > 0) {
      // Save current selection before moving
      if (selectedResponse !== null) {
        saveSelection();
      }
      
      setCurrentPairIndex(currentPairIndex - 1);
      const previousEvaluation = evaluations.get(currentPairIndex - 1);
      setSelectedResponse(previousEvaluation?.selectedIndex ?? null);
      setNote(previousEvaluation?.note ?? '');
    }
  };

  const goToNextPair = () => {
    if (currentPairIndex < data.length - 1) {
      // Save current selection before moving
      if (selectedResponse !== null) {
        saveSelection();
      }
      
      setCurrentPairIndex(currentPairIndex + 1);
      const nextEvaluation = evaluations.get(currentPairIndex + 1);
      setSelectedResponse(nextEvaluation?.selectedIndex ?? null);
      setNote(nextEvaluation?.note ?? '');
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>LLM Response Comparison Tool</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div>
              <p className="text-sm text-gray-500 mb-2">
                Upload a CSV or TSV file with: prompt, response1, response2
              </p>
              <input 
                type="file" 
                accept=".csv,.txt,.tsv"
                onChange={handleFileUpload}
                className="mb-4"
              />
            </div>
            {data.length > 0 && (
              <div className="flex justify-between items-center">
                <p className="text-sm text-green-600">
                  Loaded {data.length} comparison pairs successfully
                </p>
                <Button
                  onClick={exportToCSV}
                  disabled={evaluations.size === 0}
                >
                  <Save className="mr-2 h-4 w-4" />
                  Export All Evaluations
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {data.length > 0 && data[currentPairIndex] && (
        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Prompt</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="whitespace-pre-wrap">
                {data[currentPairIndex].prompt}
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-2 gap-4">
            <Card 
              className={`cursor-pointer transition-colors ${
                selectedResponse === 0 ? 'border-blue-500 border-2' : ''
              }`}
              onClick={() => setSelectedResponse(0)}
            >
              <CardHeader>
                <CardTitle>Response 1</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="whitespace-pre-wrap">
                  {data[currentPairIndex].response1}
                </div>
              </CardContent>
            </Card>

            <Card 
              className={`cursor-pointer transition-colors ${
                selectedResponse === 1 ? 'border-blue-500 border-2' : ''
              }`}
              onClick={() => setSelectedResponse(1)}
            >
              <CardHeader>
                <CardTitle>Response 2</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="whitespace-pre-wrap">
                  {data[currentPairIndex].response2}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardContent className="pt-6">
              <Textarea
                placeholder="Add notes about your selection..."
                value={note}
                onChange={(e) => setNote(e.target.value)}
                className="mb-4"
                rows={4}
              />

              <div className="flex justify-between items-center">
                <Button
                  variant="outline"
                  onClick={goToPreviousPair}
                  disabled={currentPairIndex === 0}
                >
                  <ChevronLeft className="mr-2 h-4 w-4" />
                  Previous
                </Button>

                <Button
                  onClick={saveSelection}
                  disabled={selectedResponse === null}
                >
                  <Save className="mr-2 h-4 w-4" />
                  Save Selection
                </Button>

                <Button
                  variant="outline"
                  onClick={goToNextPair}
                  disabled={currentPairIndex >= data.length - 1}
                >
                  Next
                  <ChevronRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-2">
                <AlertCircle className="h-4 w-4 text-blue-500" />
                <span>
                  Pair {currentPairIndex + 1} of {data.length}
                </span>
                <span className="ml-4">
                  Selections saved: {evaluations.size}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};