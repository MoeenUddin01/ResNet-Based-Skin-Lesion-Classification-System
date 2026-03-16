export interface PredictionResponse {
    class_name: string;
    confidence: number;
    probabilities: Record<string, number>;
}

export interface LesionClassInfo {
    code: string;
    name: string;
    description: string;
    color: string;
}

export const LESION_CLASSES: Record<string, LesionClassInfo> = {
    nv: {
        code: "nv",
        name: "Melanocytic Nevus",
        description: "Benign skin lesion composed of melanocytes",
        color: "#22d3ee",
    },
    mel: {
        code: "mel",
        name: "Melanoma",
        description: "Malignant tumor of melanocytes — most dangerous skin cancer",
        color: "#ef4444",
    },
    bkl: {
        code: "bkl",
        name: "Benign Keratosis",
        description: "Non-cancerous skin growth including seborrheic keratosis",
        color: "#f59e0b",
    },
    bcc: {
        code: "bcc",
        name: "Basal Cell Carcinoma",
        description: "Most common type of skin cancer — slow-growing",
        color: "#f97316",
    },
    akiec: {
        code: "akiec",
        name: "Actinic Keratosis",
        description: "Rough scaly patch from years of sun exposure",
        color: "#a78bfa",
    },
    vasc: {
        code: "vasc",
        name: "Vascular Lesion",
        description: "Lesions of blood vessel origin",
        color: "#ec4899",
    },
    df: {
        code: "df",
        name: "Dermatofibroma",
        description: "Benign fibrous nodule usually found in the skin",
        color: "#34d399",
    },
};

export function getLesionInfo(className: string): LesionClassInfo {
    const key = className.toLowerCase();
    return (
        LESION_CLASSES[key] ?? {
            code: key,
            name: className,
            description: "Skin lesion classification result",
            color: "#22d3ee",
        }
    );
}
