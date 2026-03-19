import type { NextConfig } from "next";

// Local dev:  set NEXT_PUBLIC_API_URL in frontend/.env.local
// Production: set it as a Vercel environment variable pointing to your
//             Hugging Face Spaces URL, e.g.
//             https://YOUR_USERNAME-skin-lesion-api.hf.space
const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${API_URL}/api/v1/:path*`,
      },
    ];
  },
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost" },
      { protocol: "https", hostname: "*.hf.space" },
    ],
  },
};

export default nextConfig;
