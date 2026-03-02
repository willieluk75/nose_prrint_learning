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
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-white border-b px-6 py-4 flex justify-between items-center z-10">
          <h2 className="text-xl font-bold text-gray-900">裁切鼻紋區域</h2>
          <button
            onClick={onCancel}
            disabled={isProcessing}
            className="p-2 text-gray-500 hover:text-gray-700 disabled:opacity-50 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12" />
            </svg>
          </button>
        </div>

        <div className="p-6">
          <p className="text-sm text-gray-600 mb-4">
            拖拽裁切框選取鼻紋區域（保持正方形）
          </p>

          <div className="bg-gray-100 rounded-xl overflow-hidden">
            {initialImageSrc && (
              <ReactCrop
                crop={crop}
                onChange={(c) => setCrop(c)}
                onComplete={(c) => setCompletedCrop(c)}
                aspect={1}
                minWidth={224}
                minHeight={224}
                keepSelection
              >
                <img
                  ref={imgRef}
                  alt="Crop preview"
                  src={initialImageSrc}
                  onLoad={onImageLoad}
                  className="max-w-full"
                  style={{ maxHeight: '60vh' }}
                />
              </ReactCrop>
            )}

            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6">
              <div className="flex items-start">
                <svg
                  className="w-5 h-5 text-blue-500 mt-0.5 mr-2 flex-shrink-0"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 1112 1 0 011.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
                <div className="flex-1">
                  <p className="text-sm text-blue-800 font-medium">
                    請確保裁切區域包含完整的鼻紋圖案。
                  </p>
                  <p className="text-xs text-blue-700 mt-1">
                    建議以鼻尖為中心，包含鼻子周圍的紋理。
                  </p>
                </div>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={handleCrop}
                disabled={isProcessing || !completedCrop}
                className="flex-1 py-3 px-4 bg-primary text-white rounded-xl
                  hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed
                  font-medium transition-colors"
              >
                {isProcessing ? (
                  <>
                    <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth={4}
                      />
                    </svg>
                    <span className="ml-2">處理中...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 16l4.586-4.586a2 2 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                      />
                    </svg>
                    <span className="ml-2">確認裁切</span>
                  </>
                )}
              </button>
              <button
                onClick={onCancel}
                disabled={isProcessing}
                className="py-3 px-4 border border-gray-300 text-gray-700 rounded-xl
                  hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed
                  font-medium transition-colors"
              >
                取消
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
