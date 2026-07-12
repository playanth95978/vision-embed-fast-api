import { ActionBarPrimitive, ComposerPrimitive, MessagePrimitive, ThreadPrimitive, } from "@assistant-ui/react";
import type { FC } from "react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { SendHorizontalIcon } from "lucide-react";

export const Thread: FC = () => {
  return (
    <ThreadPrimitive.Root className="bg-background h-full flex flex-col">
      <ThreadPrimitive.Viewport className="flex-grow overflow-y-auto px-4 py-6">
        <ThreadPrimitive.Empty>
          <div className="flex flex-col items-center justify-center h-full text-center">
            <h1 className="text-2xl font-bold">Comment puis-je vous aider aujourd'hui ?</h1>
          </div>
        </ThreadPrimitive.Empty>

        <ThreadPrimitive.Messages
          components={{
            UserMessage,
            AssistantMessage,
          }}
        />
      </ThreadPrimitive.Viewport>

      <div className="border-t p-4">
        <Composer />
      </div>
    </ThreadPrimitive.Root>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="mb-6 flex flex-col items-end gap-2">
      <div className="bg-primary text-primary-foreground max-w-[80%] rounded-2xl px-4 py-2">
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  );
};

const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="mb-6 flex gap-3">
      <Avatar className="h-8 w-8">
        <AvatarImage src="/assistant-avatar.png" />
        <AvatarFallback>AI</AvatarFallback>
      </Avatar>
      <div className="flex flex-col gap-2 max-w-[80%]">
        <div className="bg-muted rounded-2xl px-4 py-2">
          <MessagePrimitive.Content />
        </div>
        <ActionBar />
      </div>
    </MessagePrimitive.Root>
  );
};

const ActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root className="flex gap-2">
      <ActionBarPrimitive.Copy asChild>
        <Button variant="ghost" size="icon" className="h-6 w-6">
          <span className="sr-only">Copier</span>
        </Button>
      </ActionBarPrimitive.Copy>
      <ActionBarPrimitive.Reload asChild>
        <Button variant="ghost" size="icon" className="h-6 w-6">
          <span className="sr-only">Régénérer</span>
        </Button>
      </ActionBarPrimitive.Reload>
    </ActionBarPrimitive.Root>
  );
};

const Composer: FC = () => {
  return (
    <ComposerPrimitive.Root className="flex items-end gap-2">
      <ComposerPrimitive.Input
        autoFocus
        placeholder="Écrivez un message..."
        className="bg-muted max-h-40 flex-grow resize-none rounded-xl p-3 focus:outline-none"
      />
      <ComposerPrimitive.Send asChild>
        <Button size="icon" className="rounded-xl h-12 w-12">
          <SendHorizontalIcon className="h-5 w-5" />
        </Button>
      </ComposerPrimitive.Send>
    </ComposerPrimitive.Root>
  );
};
