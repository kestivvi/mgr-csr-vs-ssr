import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  compress: false,
  compiler: {
    removeConsole: true,
  },
  experimental: {
    optimizePackageImports: ['react', 'react-dom'],
  },
  trailingSlash: false,
  generateEtags: true,
};

export default nextConfig;
