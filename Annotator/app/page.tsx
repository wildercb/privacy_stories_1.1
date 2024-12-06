// app/page.tsx
"use client"
import { ComparisonInterface } from "@/components/comparison-interface"

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <ComparisonInterface />
    </main>
  )
}