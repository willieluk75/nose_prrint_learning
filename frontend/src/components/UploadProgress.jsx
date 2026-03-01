import { useEffect } from 'react';

/**
 * UploadProgress - Display upload progress for multiple items
 */
export function UploadProgress({ uploads, onCancel, onComplete }) {
  const allCompleted = uploads.every((u) => u.status === 'success' || u.status === 'error');
  const hasErrors = uploads.some((u) => u.status === 'error');

  useEffect(() => {
    if (allCompleted && uploads.length > 0) {
      onComplete?.();
    }
  }, [allCompleted, uploads.length, onComplete]);

  const completedCount = uploads.filter((u) => u.status === 'success').length;
  const errorCount = uploads.filter((u) => u.status === 'error').length;

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-md w-full p-6">
        <h2 className="text-xl font-bold mb-4">上傳進度</h2>

        <div className="mb-4">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>
              {completedCount} / {uploads.length} 已完成
            </span>
            {errorCount > 0 && (
              <span className="text-red-500">{errorCount} 個失敗</span>
            )}
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
            <div
              className="bg-primary h-full transition-all duration-300"
              style={{ width: `${(completedCount / uploads.length) * 100}%` }}
            />
          </div>
        </div>

        <div className="space-y-3 max-h-[300px] overflow-y-auto mb-6">
          {uploads.map((upload) => (
            <UploadItem key={upload.id} upload={upload} />
          ))}
        </div>

        {allCompleted && (
          <button
            onClick={onCancel}
            className="w-full py-3 px-4 bg-primary text-white rounded-xl
              hover:bg-primary/90 font-medium transition-colors"
          >
            完成
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * UploadItem - Individual upload status
 */
function UploadItem({ upload }) {
  const statusConfig = {
    pending: {
      icon: (
        <div className="w-5 h-5 border-2 border-gray-300 border-t-primary rounded-full animate-spin" />
      ),
      color: 'text-gray-500',
    },
    uploading: {
      icon: (
        <div className="w-5 h-5 border-2 border-gray-300 border-t-primary rounded-full animate-spin" />
      ),
      color: 'text-primary',
    },
    success: {
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      ),
      color: 'text-green-500',
    },
    error: {
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      ),
      color: 'text-red-500',
    },
  };

  const config = statusConfig[upload.status];

  return (
    <div className={`flex items-center gap-3 p-3 rounded-xl border ${
      upload.status === 'error' ? 'border-red-200 bg-red-50' : 'border-gray-100'
    }`}>
      <div className={config.color}>{config.icon}</div>
      <div className="flex-1 min-w-0">
        <p className="font-medium truncate">{upload.petName}</p>
        <p className="text-sm text-gray-500 truncate">{upload.species}</p>
      </div>
      {upload.status === 'error' && (
        <p className="text-xs text-red-500 text-right max-w-[150px]">{upload.error}</p>
      )}
    </div>
  );
}
