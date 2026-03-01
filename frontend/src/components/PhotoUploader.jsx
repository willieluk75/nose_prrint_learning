import { useState, useRef } from 'react';
import { toast } from 'react-toastify';

/**
 * PhotoUploader - Multiple photo selection with thumbnail preview
 * Supports HEIC, JPEG, PNG formats
 */
export function PhotoUploader({ onPhotosSelect, maxPhotos = 10 }) {
  const [photos, setPhotos] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

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

      // Create preview URL
      let previewUrl;
      if (file.type.includes('heic') || file.type.includes('heif')) {
        // HEIC files need to be converted, store file only for now
        previewUrl = null;
      } else {
        previewUrl = URL.createObjectURL(file);
      }

      newPhotos.push({
        id: crypto.randomUUID(),
        file,
        previewUrl,
        isHeic: file.type.includes('heic') || file.type.includes('heif'),
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
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
        <p className="text-lg font-medium text-gray-700 mb-2">
          點擊或拖拽照片到這裡
        </p>
        <p className="text-sm text-gray-500">
          支援 HEIC、JPEG、PNG 格式 (最多 {maxPhotos} 張)
        </p>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/jpg,image/png,image/heic,image/heif"
          multiple
          onChange={handleInputChange}
          className="hidden"
        />
      </div>

      {photos.length > 0 && (
        <div className="grid grid-cols-3 gap-3">
          {photos.map((photo) => (
            <div key={photo.id} className="relative group">
              {photo.previewUrl ? (
                <img
                  src={photo.previewUrl}
                  alt={`Photo ${photo.id}`}
                  className="w-full aspect-square object-cover rounded-xl shadow-md"
                />
              ) : (
                <div className="w-full aspect-square bg-gray-100 rounded-xl flex items-center justify-center">
                  <div className="text-center">
                    <span className="text-2xl">📸</span>
                    <p className="text-xs text-gray-500 mt-1">HEIC</p>
                  </div>
                </div>
              )}
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
              {photo.isHeic && (
                <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
                  HEIC
                </div>
              )}
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
