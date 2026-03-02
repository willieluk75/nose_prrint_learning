import { useState, useRef } from 'react';
import ReactCrop, { centerCrop, makeAspectCrop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';

/**
 * Convert crop to canvas and export as blob
 */
function getCroppedImg(image, crop, fileName) {
  const canvas = document.createElement('canvas');
  const scaleX = image.naturalWidth / image.width;
  const scaleY = image.naturalHeight / image.height;

  canvas.width = crop.width * scaleX;
  canvas.height = crop.height * scaleY;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(
    image,
    crop.x * scaleX,
    crop.y * scaleY,
    crop.width * scaleX,
    crop.height * scaleY
  );

  return new Promise((resolve) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          console.error('Canvas is empty');
          return;
        }
        blob.name = `${fileName.replace(/\.[^/.]+$/, '')}_cropped.jpg`;
        resolve(blob);
      },
      'image/jpeg',
      0.95
    );
  });
}

/**
 * PhotoCropper - Crop pet nose print photos with fixed aspect ratio
 * Supports croppedBlob from PhotoUploader for HEIC-converted images
 */
export function PhotoCropper({ photo, onCropComplete, onCancel }) {
  const [crop, setCrop] = useState(null);
  const [completedCrop, setCompletedCrop] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const imgRef = useRef(null);

  const initialImageSrc = photo.previewUrl
    || (photo.imageBlob ? URL.createObjectURL(photo.imageBlob) : URL.createObjectURL(photo.file));

  const onImageLoad = (e) => {
    const { width, height } = e.currentTarget;

    // Initial crop: centered square, max 800px
    const initialCrop = centerCrop(
      makeAspectCrop(
        {
          unit: '%',
          width: Math.min(80, (width / width) * 100),
        },
        1, // aspect ratio 1:1 (square)
        width,
        height
      )
    );

    setCrop(initialCrop);
  };

  const handleCrop = async () => {
    if (!imgRef.current || !completedCrop) {
      return;
    }

    setIsProcessing(true);

    try {
      const croppedBlob = await getCroppedImg(
        imgRef.current,
        completedCrop,
        photo.file.name
      );

      onCropComplete(
        new File([croppedBlob], `${photo.file.name.replace(/\.[^/.]+$/, '')}_cropped.jpg`, {
          type: 'image/jpeg',
        }),
        croppedBlob
      );
    } catch (error) {
      console.error('Error cropping image:', error);
      alert('裁切圖片時發生錯誤，請重試');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b px-4 py-3 flex items-center justify-between shrink-0">
        <h2 className="text-lg font-bold text-gray-900">裁切鼻紋區域</h2>
        <button
          onClick={onCancel}
          disabled={isProcessing}
          className="p-2 text-gray-500 hover:text-gray-700 disabled:opacity-50 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Image area — fills all available space */}
      <div className="flex-1 overflow-hidden flex items-center justify-center bg-gray-900 min-h-0">
        {initialImageSrc && (
          <ReactCrop
            crop={crop}
            onChange={(c) => setCrop(c)}
            onComplete={(c) => setCompletedCrop(c)}
            aspect={1}
            minWidth={100}
            minHeight={100}
            keepSelection
          >
            <img
              ref={imgRef}
              alt="Crop preview"
              src={initialImageSrc}
              onLoad={onImageLoad}
              style={{ maxWidth: '100vw', maxHeight: 'calc(100vh - 120px)', objectFit: 'contain' }}
            />
          </ReactCrop>
        )}
      </div>

      {/* Bottom bar */}
      <div className="bg-white border-t px-4 py-3 flex items-center gap-3 shrink-0">
        <p className="text-xs text-gray-500 flex-1">拖拽裁切框選取鼻紋區域（正方形）</p>
        <button
          onClick={onCancel}
          disabled={isProcessing}
          className="py-2 px-4 border border-gray-300 text-gray-700 rounded-xl
            hover:bg-gray-50 disabled:opacity-50 font-medium transition-colors"
        >
          取消
        </button>
        <button
          onClick={handleCrop}
          disabled={isProcessing || !completedCrop}
          className="py-2 px-5 bg-primary text-white rounded-xl
            hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed
            font-medium transition-colors flex items-center gap-2"
        >
          {isProcessing ? (
            <>
              <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth={4} />
              </svg>
              處理中...
            </>
          ) : '確認裁切'}
        </button>
      </div>
    </div>
  );
}
