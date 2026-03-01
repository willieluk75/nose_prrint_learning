import { useState, useRef, useEffect } from 'react';
import ReactCrop, { centerCrop, makeAspectCrop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';

/**
 * Convert crop to a canvas
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
    crop.height * scaleY,
    0,
    0,
    canvas.width,
    canvas.height
  );

  // Convert to blob
  return new Promise((resolve) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          console.error('Canvas is empty');
          return;
        }
        blob.name = fileName || 'cropped-image.jpg';
        resolve(blob);
      },
      'image/jpeg',
      0.95
    );
  });
}

/**
 * PhotoCropper - Crop pet nose print photos with fixed aspect ratio
 */
export function PhotoCropper({ photo, onCropComplete, onCancel }) {
  const [crop, setCrop] = useState(null);
  const [completedCrop, setCompletedCrop] = useState(null);
  const [imageSrc, setImageSrc] = useState('');
  const imgRef = useRef(null);

  useEffect(() => {
    // Load image from file
    const reader = new FileReader();
    reader.addEventListener('load', () => {
      setImageSrc(reader.result);
    });
    reader.readAsDataURL(photo.file);
  }, [photo.file]);

  const onImageLoad = (e) => {
    const { width, height } = e.currentTarget;

    // Initial crop: centered square, max size 800px
    const cropWidth = Math.min(800, width * 0.8);
    const cropHeight = cropWidth;

    const initialCrop = centerCrop(
      makeAspectCrop(
        {
          unit: '%',
          width: (cropWidth / width) * 100,
        },
        1, // aspect ratio
        width,
        height
      ),
      width,
      height
    );

    setCrop(initialCrop);
  };

  const handleCrop = async () => {
    if (!imgRef.current || !completedCrop) {
      return;
    }

    try {
      const croppedBlob = await getCroppedImg(
        imgRef.current,
        completedCrop,
        `${photo.file.name.replace(/\.[^/.]+$/, '')}_cropped.jpg`
      );

      // Convert blob to File
      const croppedFile = new File([croppedBlob], `${photo.file.name.replace(/\.[^/.]+$/, '')}_cropped.jpg`, {
        type: 'image/jpeg',
        lastModified: Date.now(),
      });

      onCropComplete(croppedFile, croppedBlob);
    } catch (error) {
      console.error('Error cropping image:', error);
      alert('裁切圖片時發生錯誤，請重試');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-white border-b px-6 py-4 flex justify-between items-center z-10">
          <h2 className="text-xl font-bold">裁切鼻紋區域</h2>
          <button
            onClick={onCancel}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6">
          <div className="mb-4">
            <p className="text-sm text-gray-600 mb-2">
              拖拽裁切框選取鼻紋區域，保持正方形
            </p>
            <div className="bg-gray-100 rounded-xl overflow-hidden">
              {imageSrc && (
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
                    src={imageSrc}
                    onLoad={onImageLoad}
                    className="max-w-full h-auto"
                  />
                </ReactCrop>
              )}
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6">
            <div className="flex items-start">
              <svg
                className="w-5 h-5 text-blue-500 mt-0.5 mr-2 flex-shrink-0"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
              <p className="text-sm text-blue-800">
                請確保裁切區域包含完整的鼻紋圖案。建議以鼻尖為中心，包含鼻子周圍的紋理。
              </p>
            </div>
          </div>

          <div className="flex gap-3">
            <button
              onClick={onCancel}
              className="flex-1 py-3 px-4 border border-gray-300 rounded-xl
                hover:bg-gray-50 font-medium transition-colors"
            >
              取消
            </button>
            <button
              onClick={handleCrop}
              disabled={!completedCrop}
              className="flex-1 py-3 px-4 bg-primary text-white rounded-xl
                hover:bg-primary/90 font-medium transition-colors
                disabled:opacity-50 disabled:cursor-not-allowed"
            >
              確認裁切
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
