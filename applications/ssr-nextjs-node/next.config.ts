import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  poweredByHeader: false,
  compress: false,
  productionBrowserSourceMaps: false,

  compiler: {
    removeConsole: { exclude: ['error', 'warn'] },
  },
}

export default nextConfig