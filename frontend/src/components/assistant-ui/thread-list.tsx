import {
  ThreadListPrimitive,
} from "@assistant-ui/react";
import type { FC } from "react";
import { Button } from "@/components/ui/button";
import { PlusIcon } from "lucide-react";

export const ThreadList: FC = () => {
  return (
    <ThreadListPrimitive.Root className="flex flex-col h-full border-r w-64">
      <div className="p-4 border-b">
        <ThreadListPrimitive.New asChild>
          <Button variant="outline" className="w-full flex gap-2">
            <PlusIcon className="h-4 w-4" />
            Nouvelle Conversation
          </Button>
        </ThreadListPrimitive.New>
      </div>
      <div className="flex-grow overflow-y-auto p-2">
        <ThreadListPrimitive.Items
          components={{
            ThreadListItem,
          }}
        />
      </div>
    </ThreadListPrimitive.Root>
  );
};

const ThreadListItem: FC = () => {
  return (
    <ThreadListPrimitive.Item className="mb-1 rounded-lg hover:bg-muted p-2 cursor-pointer focus:outline-none data-active:bg-muted">
      <ThreadListPrimitive.ItemTrigger className="w-full text-left">
        <ThreadListPrimitive.ItemTitle className="text-sm font-medium truncate" />
      </ThreadListPrimitive.ItemTrigger>
    </ThreadListPrimitive.Item>
  );
};
