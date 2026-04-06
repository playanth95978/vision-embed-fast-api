import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Upload, X } from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"

// Since we added the endpoint, the generated client might not have it yet.
// We'll use a direct fetch or assume the client will be regenerated.
// For now, let's use a manual fetch for the multipart form data.
const uploadImage = async ({ file, description }: { file: File, description: string }) => {
  const formData = new FormData()
  formData.append("file", file)
  formData.append("description", description)

  const token = localStorage.getItem("access_token")
  const response = await fetch("/api/v1/images/", {
    method: "POST",
    headers: {
        Authorization: `Bearer ${token}`,
    },
    body: formData,
  })

  if (!response.ok) {
    throw new Error("Upload failed")
  }

  return response.json()
}

export function ImageUploadCard() {
  const [file, setFile] = useState<File | null>(null)
  const [description, setDescription] = useState("")
  const [preview, setPreview] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: uploadImage,
    onSuccess: () => {
      toast.success("Image uploaded successfully!")
      setFile(null)
      setDescription("")
      setPreview(null)
      queryClient.invalidateQueries({ queryKey: ["images"] })
    },
    onError: (error) => {
      toast.error(`Error: ${error.message}`)
    },
  })

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result as string)
      }
      reader.readAsDataURL(selectedFile)
    }
  }

  const handleUpload = () => {
    if (file) {
      mutation.mutate({ file, description })
    }
  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Upload Image</CardTitle>
        <CardDescription>
          Upload an image and provide a description. We will generate a vector embedding for it.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="image">Image</Label>
          <div className="flex flex-col items-center justify-center border-2 border-dashed border-muted rounded-lg p-6 hover:border-primary transition-colors">
            {preview ? (
              <div className="relative w-full aspect-video">
                <img src={preview} alt="Preview" className="w-full h-full object-contain rounded-md" />
                <Button 
                  variant="destructive" 
                  size="icon" 
                  className="absolute top-2 right-2 rounded-full h-8 w-8"
                  onClick={() => {
                    setFile(null)
                    setPreview(null)
                  }}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2">
                <Upload className="h-10 w-10 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">Click to upload or drag and drop</p>
                <Input 
                  id="image" 
                  type="file" 
                  accept="image/*" 
                  className="hidden" 
                  onChange={handleFileChange}
                />
                <Button variant="outline" onClick={() => document.getElementById("image")?.click()}>
                  Select File
                </Button>
              </div>
            )}
          </div>
        </div>
        <div className="space-y-2">
          <Label htmlFor="description">Description (Stored with vector embedding)</Label>
          <Textarea 
            id="description" 
            placeholder="Describe the image content..." 
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>
      </CardContent>
      <CardFooter>
        <Button 
          className="w-full" 
          disabled={!file || mutation.isPending}
          onClick={handleUpload}
        >
          {mutation.isPending ? "Uploading..." : "Upload and Save Vector Data"}
        </Button>
      </CardFooter>
    </Card>
  )
}
