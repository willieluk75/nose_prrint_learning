import { useRef, useState } from 'react';
import { toast } from 'react-toastify';
import heic2any from 'heic2any';

export function PhotoUploader({ photos, onPhotosChange, onPhotoClick, maxPhotos = 10 }) {
  const [isDragging, setIsDragging] = useState(false);
  const [converting, setConverting] = useState(null);
  const fileInputRef = useRef(null);

  const convertHeicToJpeg = async (file) => {
    setConverting(file.name);
    try {
      const blob = await heic2any({ blob: file, toType: 'image/jpeg', quality: 0.92 });
      return new File([blob], `${file.name.replace(/\.[^/.]+$/, '')}.jpg`, { type: 'image/jpeg' });
    } catch (err) {
      throw new Error(`HEIC 轉換失敗: ${err.message}`);
    } finally {
      setConverting(null);
    }
  };

  const handleFileSelect = async (files) => {
    const newPhotos = [];
    for (const file of Array.from(files)) {
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/heic', 'image/heif'];
      if (!validTypes.includes(file.type)) {
        toast.error(`不支援的格式: ${file.name}`);
        continue;
      }
      if (photos.length + newPhotos.length >= maxPhotos) {
        toast.error(`最多只能選擇 ${maxPhotos} 張照片`);
        break;
      }
      let finalFile = file;
      if (file.type.includes('heic') || file.type.includes('heif')) {
        try {
          finalFile = await convertHeicToJpeg(file);
        } catch (err) {
          toast.error(err.message);
          continue;
        }
      }
      newPhotos.push({
        id: crypto.randomUUID(),
        file: finalFile,
        previewUrl: URL.createObjectURL(finalFile),
      });
    }
    if (newPhotos.length > 0) {
      onPhotosChange([...photos, ...newPhotos]);
    }
  };

  const removePhoto = (e, photoId) => {
    e.stopPropagation();
    onPhotosChange(photos.filter((p) => p.id !== photoId));
  };

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
        onDrop={(e) => { e.preventDefault(); setIsDragging(false); handleFileSelect(e.dataTransfer.files); }}
        className={`border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all min-h-[160px] ${
          isDragging
            ? 'border-primary bg-primary/5 scale-105'
            : 'border-gray-300 hover:border-primary hover:bg-gray-50'
        }`}
      >
        <svg className="w-12 h-12 text-gray-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <p className="text-base font-medium text-gray-700">點擊或拖拽照片到這裡</p>
        <p className="text-sm text-gray-500 mt-1">
          支援 HEIC、JPEG、PNG（最多 {maxPhotos} 張）
          {converting && <span className="text-primary ml-1">轉換中: {converting}...</span>}
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/jpg,image/png,image/heic,image/heif"
          multiple
          onChange={(e) => handleFileSelect(e.target.files)}
          disabled={converting !== null}
          className="hidden"
        />
      </div>

      {/* Photo grid — controlled by parent, shows latest state including crops */}
      {photos.length > 0 && (
        <>
          <div className="grid grid-cols-3 gap-3">
            {photos.map((photo) => (
              <div
                key={photo.id}
                className="relative group cursor-pointer"
                onClick={() => onPhotoClick?.(photo)}
              >
                <img
                  src={photo.previewUrl}
                  alt=""
                  className="w-full aspect-square object-cover rounded-xl shadow-md group-hover:ring-2 group-hover:ring-primary transition-all"
                />
                <button
                  onClick={(e) => removePhoto(e, photo.id)}
                  className="absolute top-1.5 right-1.5 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center shadow opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
          <div className="flex justify-between items-center text-sm text-gray-500">
            <span>已選擇 {photos.length} 張 · 點擊照片可裁切</span>
            <button
              onClick={() => { if (confirm('確定要清空所有照片嗎？')) onPhotosChange([]); }}
              className="text-red-500 hover:text-red-700 font-medium"
            >
              清空
            </button>
          </div>
        </>
      )}
    </div>
  );
}
