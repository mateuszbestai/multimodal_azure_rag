import React, { useState, useRef } from 'react';
import { Upload, FileSpreadsheet, FileText, Image as ImageIcon, Loader2, CheckCircle, AlertCircle, X, Database, Table, BarChart3 } from 'lucide-react';

interface UploadedFile {
  docId: string;
  fileName: string;
  contentType: string;
  sheetCount?: number;
}

interface UploadResult {
  filename: string;
  file_type: string;
  summary: string;
  node_count: number;
  sheet_count?: number;
  image_count?: number;
}

interface FileUploadProps {
  onUploadComplete?: (result: UploadResult) => void;
  onFilesChange?: (files: UploadedFile[]) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUploadComplete, onFilesChange }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load uploaded files on component mount
  React.useEffect(() => {
    loadUploadedFiles();
  }, []);

  const loadUploadedFiles = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/files');
      if (response.ok) {
        const data = await response.json();
        setUploadedFiles(data.files);
        onFilesChange?.(data.files);
      }
    } catch (error) {
      console.error('Error loading files:', error);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setError(null);
    setUploadResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:5001/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const result = await response.json();
      setUploadResult(result);
      onUploadComplete?.(result);
      
      // Reload the files list
      await loadUploadedFiles();
      
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Upload failed');
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const deleteFile = async (docId: string) => {
    try {
      const response = await fetch(`http://localhost:5001/api/files/${docId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        await loadUploadedFiles();
      } else {
        console.error('Failed to delete file');
      }
    } catch (error) {
      console.error('Error deleting file:', error);
    }
  };

  const getFileIcon = (contentType: string) => {
    if (contentType.includes('excel')) {
      return <FileSpreadsheet className="text-green-600" size={20} />;
    } else if (contentType.includes('pdf')) {
      return <FileText className="text-red-600" size={20} />;
    }
    return <Database className="text-gray-600" size={20} />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragging
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDrop={handleDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.xlsx,.xls,.xlsm"
          onChange={handleFileSelect}
          className="hidden"
        />
        
        {uploading ? (
          <div className="space-y-4">
            <Loader2 className="mx-auto animate-spin text-primary-600" size={48} />
            <p className="text-gray-600">Processing your file...</p>
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className="mx-auto text-gray-400" size={48} />
            <div>
              <p className="text-lg font-medium text-gray-900">Upload a document</p>
              <p className="text-gray-500">Drag and drop or click to select</p>
              <p className="text-sm text-gray-400 mt-2">
                Supports: PDF, Excel (.xlsx, .xls, .xlsm) - Max 16MB
              </p>
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 transition-colors"
            >
              Choose File
            </button>
          </div>
        )}
      </div>

      {/* Upload Result */}
      {uploadResult && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="text-green-600" size={20} />
            <h3 className="font-medium text-green-800">Upload Successful</h3>
          </div>
          <div className="text-sm text-green-700 space-y-1">
            <p><strong>File:</strong> {uploadResult.filename}</p>
            <p><strong>Type:</strong> {uploadResult.file_type.toUpperCase()}</p>
            <p><strong>Summary:</strong> {uploadResult.summary}</p>
            <div className="flex gap-4 mt-2">
              <span className="flex items-center gap-1">
                <Database size={16} />
                {uploadResult.node_count} search nodes
              </span>
              {uploadResult.sheet_count && (
                <span className="flex items-center gap-1">
                  <Table size={16} />
                  {uploadResult.sheet_count} sheets
                </span>
              )}
              {uploadResult.image_count && uploadResult.image_count > 0 && (
                <span className="flex items-center gap-1">
                  <ImageIcon size={16} />
                  {uploadResult.image_count} images
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <AlertCircle className="text-red-600" size={20} />
            <h3 className="font-medium text-red-800">Upload Failed</h3>
          </div>
          <p className="text-sm text-red-700 mt-1">{error}</p>
        </div>
      )}

      {/* Uploaded Files List */}
      {uploadedFiles.length > 0 && (
        <div className="bg-white border rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b bg-gray-50">
            <h3 className="font-medium text-gray-900">Uploaded Documents</h3>
            <p className="text-sm text-gray-500">{uploadedFiles.length} files in knowledge base</p>
          </div>
          <div className="divide-y">
            {uploadedFiles.map((file) => (
              <div key={file.docId} className="px-4 py-3 flex items-center justify-between hover:bg-gray-50">
                <div className="flex items-center gap-3">
                  {getFileIcon(file.contentType)}
                  <div>
                    <p className="font-medium text-gray-900">{file.fileName}</p>
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <span className="capitalize">{file.contentType.replace('_', ' ')}</span>
                      {file.sheetCount && (
                        <>
                          <span>•</span>
                          <span>{file.sheetCount} sheets</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => deleteFile(file.docId)}
                  className="p-1 text-gray-400 hover:text-red-600 transition-colors"
                  title="Delete file"
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;