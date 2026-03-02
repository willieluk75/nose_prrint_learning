import { useState, useRef, useEffect } from 'react';
import { toast } from 'react-toastify';
import heic2any from 'heic2any';

/**
 * PhotoUploader - Multiple photo selection with thumbnail preview
 * Supports HEIC, JPEG, PNG formats with automatic HEIC conversion
 */
export function PhotoUploader({ onPhotosSelect, maxPhotos = 10 }) {
  const [photos, setPhotos] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  const [converting, setConverting] = useState(null);

  /**
   * Convert HEIC to JPEG
   */
  const convertHeicToJpeg = async (heicFile) => {
    setConverting(heicFile.name);
    try {
      const jpegBlob = await heic2any({
        blob: heicFile,
        toType: 'image/jpeg',
        quality: 0.92,
      });
      const jpegFile = new File([jpegBlob], `${heicFile.name.replace(/\.[^/.]+$/, '')}.jpg`, {
        type: 'image/jpeg',
      });
      setConverting(null);
      return jpegFile;
    } catch (error) {
      console.error('HEIC conversion error:', error);
      setConverting(null);
      throw new Error(`HEIC 轉換失敗: ${error.message}`);
    }
  };

  const handleFileSelect = async (files) => {
    const newPhotos = [];

    for (let file of Array.from(files)) {
      // Check file type
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/heic', 'image/heif'];
      if (!validTypes.includes(file.type)) {
        toast.error(`不支援的檔案格式: ${file.name}`);
        continue;
      }

      // Check if max photos reached
      if (photos.length + newPhotos.length >= maxPhotos) {
        toast.error(`最多只能選擇 ${maxPhotos} 張照片`);
        break;
      }

      let finalFile = file;
      let previewUrl;

      // Handle HEIC files
      if (file.type.includes('heic') || file.type.includes('heif')) {
        try {
          finalFile = await convertHeicToJpeg(file);
          previewUrl = URL.createObjectURL(finalFile);
        } catch (error) {
          toast.error(error.message);
          continue;
        }
      } else {
        previewUrl = URL.createObjectURL(file);
      }

      newPhotos.push({
        id: crypto.randomUUID(),
        file: finalFile,
        previewUrl,
        isHeic: false, // Already converted to JPEG
      });
    }

    if (newPhotos.length > 0) {
      const updatedPhotos = [...photos, ...newPhotos];
      setPhotos(updatedPhotos);
      onPhotosSelect(updatedPhotos);
    }
  };

  const handleInputChange = (e) => {
    handleFileSelect(e.target.files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const removePhoto = (photoId) => {
    const updatedPhotos = photos.filter((p) => p.id !== photoId);
    setPhotos(updatedPhotos);
    onPhotosSelect(updatedPhotos);
  };

  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="space-y-4">
      <div
        onClick={openFilePicker}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-2xl p-8
          flex flex-col items-center justify-center
          cursor-pointer transition-all
          min-h-[200px]
          ${isDragging
            ? 'border-primary bg-primary/5 scale-105'
            : 'border-gray-300 hover:border-primary hover:bg-gray-50'
          }
        `}
      >
        <svg
          className="w-16 h-16 text-gray-400 mb-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
          />
        </svg>
        <p className="text-lg font-medium text-gray-700 mb-2">
          點擊或拖拽照片到這裡
        </p>
        <p className="text-sm text-gray-500">
          支援 HEIC、JPEG、PNG 格式 (最多 {maxPhotos} 張)
          {converting && (
            <span className="text-primary ml-2">正在轉換 HEIC: {converting}...</span>
          )}
        </p>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/jpg,image/png,image/heic,image/heif"
          multiple
          onChange={handleInputChange}
          disabled={converting !== null}
          className="hidden"
        />
      </div>

      {photos.length > 0 && (
        <div className="grid grid-cols-3 gap-3">
          {photos.map((photo) => (
            <div key={photo.id} className="relative group">
              <img
                src={photo.previewUrl}
                alt={`Photo ${photo.id}`}
                className="w-full aspect-square object-cover rounded-xl shadow-md"
              ></img>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  removePhoto(photo.id);
                }}
                className="absolute top-2 right-2 w-7 h-7 bg-red-500 text-white rounded-full
                flex items-center justify-center shadow-lg
                opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}

      {photos.length > 0 && (
        <div className="flex justify-between items-center text-sm text-gray-600">
          <span>已選擇 {photos.length} 張照片</span>
          <button
            onClick={() => {
              if (confirm('確定要清空所有照片嗎？')) {
                setPhotos([]);
                onPhotosSelect([]);
              }
            }}
            className="text-red-500 hover:text-red-700 font-medium"
          >
            清空全部
          </button>
        </div>
      )}
    </div>
  );
}
