import { useState, useRef, useCallback } from 'react';
import ReactCrop, { centerCrop, makeAspectCrop } from 'react-image-crop';
import { toast } from 'react-toastify';
import 'react-image-crop/dist/ReactCrop.css';

async function cropToBlob(image, crop) {
  const canvas = document.createElement('canvas');
  const scaleX = image.naturalWidth / image.width;
  const scaleY = image.naturalHeight / image.height;
  canvas.width = Math.round(crop.width * scaleX);
  canvas.height = Math.round(crop.height * scaleY);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(
    image,
    crop.x * scaleX, crop.y * scaleY,
    crop.width * scaleX, crop.height * scaleY,
    0, 0, canvas.width, canvas.height
  );
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Canvas empty'))),
      'image/jpeg',
      0.95
    );
  });
}

export function PhotoCropper({ photo, onCropComplete, onCancel }) {
  const [crop, setCrop] = useState();
  const [completedCrop, setCompletedCrop] = useState();
  const [isProcessing, setIsProcessing] = useState(false);
  const imgRef = useRef(null);

  const onImageLoad = useCallback((e) => {
    const { width, height } = e.currentTarget;
    setCrop(
      centerCrop(makeAspectCrop({ unit: '%', width: 80 }, 1, width, height), width, height)
    );
  }, []);

  const handleConfirm = async () => {
    if (!imgRef.current || !completedCrop?.width) return;
    setIsProcessing(true);
    try {
      const blob = await cropToBlob(imgRef.current, completedCrop);
      const baseName = (photo.file?.name || 'photo.jpg').replace(/\.[^.]+$/, '');
      const croppedFile = new File([blob], `${baseName}_cropped.jpg`, { type: 'image/jpeg' });
      onCropComplete(croppedFile, blob);
    } catch (err) {
      console.error(err);
      toast.error('裁切失敗，請重試');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black flex flex-col">
      {/* Header */}
      <div className="bg-white h-14 px-4 flex items-center justify-between shrink-0 border-b">
        <h2 className="text-base font-bold text-gray-900">裁切照片</h2>
        <button
          onClick={onCancel}
          disabled={isProcessing}
          className="p-2 text-gray-500 hover:text-gray-700 disabled:opacity-50"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Crop area */}
      <div className="flex-1 flex items-center justify-center overflow-hidden min-h-0 bg-black">
        <ReactCrop
          crop={crop}
          onChange={setCrop}
          onComplete={setCompletedCrop}
          aspect={1}
          minWidth={80}
          keepSelection
        >
          <img
            ref={imgRef}
            src={photo.previewUrl}
            alt="crop"
            onLoad={onImageLoad}
            style={{
              maxWidth: '100vw',
              maxHeight: 'calc(100dvh - 116px)',
              objectFit: 'contain',
              display: 'block',
            }}
          />
        </ReactCrop>
      </div>

      {/* Footer */}
      <div className="bg-white h-[52px] px-4 flex items-center justify-end gap-3 shrink-0 border-t">
        <button
          onClick={onCancel}
          disabled={isProcessing}
          className="py-2 px-5 rounded-xl border border-gray-300 text-gray-700 text-sm font-medium hover:bg-gray-50 disabled:opacity-50"
        >
          取消
        </button>
        <button
          onClick={handleConfirm}
          disabled={isProcessing || !completedCrop?.width}
          className="py-2 px-5 rounded-xl bg-primary text-white text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {isProcessing ? (
            <>
              <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth={4} />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              處理中...
            </>
          ) : '確定'}
        </button>
      </div>
    </div>
  );
}
