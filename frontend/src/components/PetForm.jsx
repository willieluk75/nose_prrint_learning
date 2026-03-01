import { useState, useEffect } from 'react';

/**
 * PetForm - Form for entering pet information
 */
export function PetForm({ onSubmit, initialData = {}, submitLabel = '上傳' }) {
  const [formData, setFormData] = useState({
    name: '',
    species: '',
    breed: '',
    notes: '',
    ...initialData,
  });

  const [errors, setErrors] = useState({});

  const validate = () => {
    const newErrors = {};

    if (!formData.name.trim()) {
      newErrors.name = '請輸入寵物姓名';
    }

    if (!formData.species) {
      newErrors.species = '請選擇物種';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (validate()) {
      onSubmit(formData);
    }
  };

  const handleInputChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));

    // Clear error when user starts typing
    if (errors[field]) {
      setErrors((prev) => ({
        ...prev,
        [field]: null,
      }));
    }
  };

  // Update form when initialData changes
  useEffect(() => {
    setFormData((prev) => ({
      ...prev,
      ...initialData,
    }));
  }, [initialData]);

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {/* 寵物姓名 */}
      <div>
        <label htmlFor="pet-name" className="block text-sm font-medium text-gray-700 mb-2">
          寵物姓名 <span className="text-red-500">*</span>
        </label>
        <input
          id="pet-name"
          type="text"
          value={formData.name}
          onChange={(e) => handleInputChange('name', e.target.value)}
          placeholder="例如：小白、喵喵"
          className={`w-full px-4 py-3 rounded-xl border-2
            focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all
            ${errors.name ? 'border-red-500' : 'border-gray-200 focus:border-primary'}`}
        />
        {errors.name && (
          <p className="mt-1 text-sm text-red-500">{errors.name}</p>
        )}
      </div>

      {/* 物種 */}
      <div>
        <label htmlFor="pet-species" className="block text-sm font-medium text-gray-700 mb-2">
          物種 <span className="text-red-500">*</span>
        </label>
        <select
          id="pet-species"
          value={formData.species}
          onChange={(e) => handleInputChange('species', e.target.value)}
          className={`w-full px-4 py-3 rounded-xl border-2 bg-white
            focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all
            ${errors.species ? 'border-red-500' : 'border-gray-200 focus:border-primary'}`}
        >
          <option value="">請選擇物種</option>
          <option value="dog">狗</option>
          <option value="cat">貓</option>
        </select>
        {errors.species && (
          <p className="mt-1 text-sm text-red-500">{errors.species}</p>
        )}
      </div>

      {/* 品種 */}
      <div>
        <label htmlFor="pet-breed" className="block text-sm font-medium text-gray-700 mb-2">
          品種 <span className="text-gray-400">(選填)</span>
        </label>
        <input
          id="pet-breed"
          type="text"
          value={formData.breed}
          onChange={(e) => handleInputChange('breed', e.target.value)}
          placeholder="例如：柴犬、英短"
          className="w-full px-4 py-3 rounded-xl border-2 border-gray-200
            focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all
            focus:border-primary"
        />
      </div>

      {/* 備註 */}
      <div>
        <label htmlFor="pet-notes" className="block text-sm font-medium text-gray-700 mb-2">
          備註 <span className="text-gray-400">(選填)</span>
        </label>
        <textarea
          id="pet-notes"
          value={formData.notes}
          onChange={(e) => handleInputChange('notes', e.target.value)}
          placeholder="例如：年齡、特徵等資訊"
          rows={3}
          className="w-full px-4 py-3 rounded-xl border-2 border-gray-200
            focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all
            focus:border-primary resize-none"
        />
      </div>

      {/* 提交按鈕 */}
      <button
        type="submit"
        className="w-full py-4 px-4 bg-primary text-white rounded-xl
          hover:bg-primary/90 active:scale-[0.98]
          font-medium text-lg transition-all shadow-lg shadow-primary/30"
      >
        {submitLabel}
      </button>
    </form>
  );
}
