import { createFileRoute } from "@tanstack/react-router"

import { ImageUploadCard } from "@/components/Common/ImageUploadCard"

export const Route = createFileRoute("/_layout/documents")({
  component: Documents,
  head: () => ({
    meta: [
      {
        title: "Documents - Embedded your documents",
      },
    ],
  }),
})

function Documents() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Image Vector Storage</h1>
        <p className="text-muted-foreground">Upload images to store them with vector embeddings in PostgreSQL</p>
      </div>
      <ImageUploadCard />
    </div>
  )
}
