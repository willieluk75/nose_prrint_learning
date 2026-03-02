import { useState, useCallback, useEffect } from 'react';
import { PhotoUploader } from './components/PhotoUploader';
import { PhotoCropper } from './components/PhotoCropper';
import { PetForm } from './components/PetForm';
import { UploadProgress } from './components/UploadProgress';
import { useUpload } from './hooks/useUpload';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

function App() {
  const [photos, setPhotos] = useState([]);
  const [editingPhoto, setEditingPhoto] = useState(null);
  const [currentPhoto, setCurrentPhoto] = useState(null);
  const [petData, setPetData] = useState({});
  const [showProgress, setShowProgress] = useState(false);
  const [appVersion, setAppVersion] = useState('1.0.0');

  const { uploads, uploadPets, clearUploads } = useUpload();

  // Read version from public/version.json
  useEffect(() => {
    fetch('/version.json')
      .then((res) => res.json())
      .then((data) => setAppVersion(data.version))
      .catch(() => console.error('Failed to load version'));
  }, []);

  const handlePhotosSelect = useCallback((selectedPhotos) => {
    setPhotos(selectedPhotos);
  }, []);

  const handleCropComplete = useCallback((croppedFile, croppedBlob) => {
    const updatedPhotos = photos.map((photo) => {
      if (photo.id === editingPhoto.id) {
        const previewUrl = URL.createObjectURL(croppedBlob);
        return { ...photo, file: croppedFile, previewUrl, isHeic: false, croppedBlob };
      }
      return photo;
    });
    setPhotos(updatedPhotos);
    setEditingPhoto(null);
    toast.success('照片裁切完成');
  }, [editingPhoto, photos]);

  const handlePhotoClick = (photo) => {
    setEditingPhoto(photo);
  };

  const handleSelectPhotoForForm = () => {
    if (photos.length === 1) {
      setCurrentPhoto(photos[0]);
    } else {
      toast.warning('請選擇一張照片上傳');
    }
  };

  const handleFormSubmit = useCallback((formData) => {
    setPetData(formData);
    const pet = {
      ...formData,
      image: currentPhoto.file,
      imageBlob: currentPhoto.croppedBlob || currentPhoto.file,
    };
    uploadPets([pet], { sequential: true });
    setShowProgress(true);
    setCurrentPhoto(null);
  }, [currentPhoto, uploadPets]);

  const handleUploadComplete = useCallback(() => {
    setPhotos([]);
    setPetData({});
    clearUploads();
  }, [clearUploads]);

  const handleProgressClose = useCallback(() => {
    setShowProgress(false);
    clearUploads();
  }, [clearUploads]);

  const photoActions = photos.length > 0 ? (
    <>
      <div className="mt-4 flex flex-wrap gap-2">
        <button
          onClick={() => setCurrentPhoto(photos[0])}
          disabled={photos.length !== 1}
          className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-colors"
        >
          繼續填寫資料 ({photos.length})
        </button>
        <button
          onClick={handleSelectPhotoForForm}
          className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 font-medium transition-colors"
        >
          {photos.length > 1 ? '請先選擇一張照片' : '填寫資料'}
        </button>
      </div>
      <p className="text-sm text-gray-500 mt-2">
        點擊照片可進行裁切調整
      </p>
    </>
  ) : null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      <ToastContainer position="top-center" autoClose={3000} limit={3} hideProgressBar={false} newestOnTop={false} closeOnClick rtl={false} pauseOnFocusLoss draggable pauseOnHover theme="colored" />
      <div className="max-w-2xl mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-primary to-purple-600 rounded-2xl shadow-lg mb-4">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
            </svg>
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            寵物鼻紋收集
          </h1>
          <p className="text-gray-600">上傳寵物照片，收集鼻紋用於身份辨識</p>
          <p className="text-sm text-gray-400 mt-1">v{appVersion}</p>
        </div>
        <div className="bg-white rounded-3xl shadow-xl p-6 space-y-6">
          <section>
            <h2 className="text-lg font-semibold text-gray-900 mb-4">1. 上傳照片</h2>
            <PhotoUploader onPhotosSelect={handlePhotosSelect} />
            {photoActions}
          </section>
          {photos.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">已選擇的照片</h2>
              <div className="grid grid-cols-3 gap-3">
                {photos.map((photo) => (
                  <div key={photo.id} onClick={() => handlePhotoClick(photo)} className="relative group cursor-pointer">
                    {photo.previewUrl ? (
                      <img src={photo.previewUrl} alt={`Photo ${photo.id}`} className="w-full aspect-square object-cover rounded-xl shadow-md group-hover:ring-2 group-hover:ring-primary transition-all" />
                    ) : (
                      <div className="w-full aspect-square bg-gray-100 rounded-xl flex items-center justify-center">
                        <div className="text-center">
                          <span className="text-2xl">📸</span>
                          <p className="text-xs text-gray-500 mt-1">HEIC</p>
                        </div>
                      </div>
                    )}
                    {photo.id === currentPhoto?.id && (
                      <div className="absolute inset-0 bg-primary/20 rounded-xl flex items-center justify-center">
                        <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414 1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}
          {currentPhoto && (
            <section className="border-t pt-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">2. 填寫寵物資料</h2>
              <PetForm onSubmit={handleFormSubmit} initialData={petData} />
            </section>
          )}
          {photos.length === 0 && !currentPhoto && uploads.length > 0 && (
            <section className="bg-green-50 border border-green-200 rounded-2xl p-6 text-center">
              <svg className="w-12 h-12 text-green-500 mx-auto mb-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414 1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <h3 className="text-lg font-semibold text-green-900 mb-2">上傳完成！</h3>
              <p className="text-green-700">寵物資料已成功上傳到資料庫</p>
            </section>
          )}
        </div>
        <footer className="text-center mt-8 text-sm text-gray-500">
          <p>寵物鼻紋辨識系統 v{appVersion}</p>
        </footer>
      </div>
      {editingPhoto && <PhotoCropper photo={editingPhoto} onCropComplete={handleCropComplete} onCancel={() => setEditingPhoto(null)} />}
      {showProgress && <UploadProgress uploads={uploads} onCancel={handleProgressClose} onComplete={handleUploadComplete} />}
    </div>
  );
}

export default App;
