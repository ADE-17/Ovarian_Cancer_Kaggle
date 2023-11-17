{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'pl' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m/home/woody/iwso/iwso092h/ucb_kaggle/train_run2.ipynb Cell 1\u001b[0m line \u001b[0;36m3\n\u001b[1;32m      <a href='vscode-notebook-cell://ssh-remote%2Bwoody.nhr.fau.de/home/woody/iwso/iwso092h/ucb_kaggle/train_run2.ipynb#W0sdnNjb2RlLXJlbW90ZQ%3D%3D?line=0'>1</a>\u001b[0m \u001b[39mimport\u001b[39;00m \u001b[39mtimm\u001b[39;00m\n\u001b[0;32m----> <a href='vscode-notebook-cell://ssh-remote%2Bwoody.nhr.fau.de/home/woody/iwso/iwso092h/ucb_kaggle/train_run2.ipynb#W0sdnNjb2RlLXJlbW90ZQ%3D%3D?line=2'>3</a>\u001b[0m \u001b[39mclass\u001b[39;00m \u001b[39mUBCModel\u001b[39;00m(pl\u001b[39m.\u001b[39mLightningModule):\n\u001b[1;32m      <a href='vscode-notebook-cell://ssh-remote%2Bwoody.nhr.fau.de/home/woody/iwso/iwso092h/ucb_kaggle/train_run2.ipynb#W0sdnNjb2RlLXJlbW90ZQ%3D%3D?line=4'>5</a>\u001b[0m     \u001b[39mdef\u001b[39;00m \u001b[39m__init__\u001b[39m(\u001b[39mself\u001b[39m, steps_per_epoch):\n\u001b[1;32m      <a href='vscode-notebook-cell://ssh-remote%2Bwoody.nhr.fau.de/home/woody/iwso/iwso092h/ucb_kaggle/train_run2.ipynb#W0sdnNjb2RlLXJlbW90ZQ%3D%3D?line=5'>6</a>\u001b[0m         \u001b[39msuper\u001b[39m(UBCModel, \u001b[39mself\u001b[39m)\u001b[39m.\u001b[39m\u001b[39m__init__\u001b[39m()\n",
      "\u001b[0;31mNameError\u001b[0m: name 'pl' is not defined"
     ]
    }
   ],
   "source": [
    "import timm\n",
    "\n",
    "class UBCModel(pl.LightningModule):\n",
    "\n",
    "    def __init__(self, steps_per_epoch):\n",
    "        super(UBCModel, self).__init__()\n",
    "        self.num_classes = 5\n",
    "        self.steps_per_epoch = steps_per_epoch\n",
    "\n",
    "        self.model = timm.create_model('tf_efficientnet_b3',\n",
    "                                       checkpoint_path='/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b3/1/tf_efficientnet_b3_aa-84b4657e.pth')\n",
    "\n",
    "        \n",
    "        self.model.classifier= torch.nn.Linear(in_features=1536, out_features=self.num_classes, bias=True)\n",
    "        self.criterion = nn.CrossEntropyLoss()\n",
    "        self.f1 = MulticlassF1Score(num_classes=self.num_classes, average='macro')\n",
    "        self.accuracy = torchmetrics.Accuracy(num_classes=self.num_classes, task='multiclass')\n",
    "        self.precision = torchmetrics.Precision(average='macro', num_classes=self.num_classes, task='multiclass')\n",
    "        self.recall = torchmetrics.Recall(average='macro', num_classes=self.num_classes, task='multiclass')\n",
    "        \n",
    "    def forward(self, x):\n",
    "        x = self.model(x)\n",
    "        return x\n",
    "\n",
    "    def training_step(self, batch, batch_idx):\n",
    "        x, y = batch\n",
    "        y_pred = self(x)\n",
    "        loss = self.criterion(y_pred, y)\n",
    "        self.log('train_loss', loss)\n",
    "        self.log('train_f1', self.f1(y_pred, y))\n",
    "        return loss\n",
    "\n",
    "    def validation_step(self, batch, batch_idx):\n",
    "        x, y = batch\n",
    "        y_pred = self(x)\n",
    "        loss = self.criterion(y_pred, y)\n",
    "        self.log('val_loss', loss)\n",
    "        self.log('val_f1', self.f1(y_pred, y))\n",
    "        self.log('val_acc', self.accuracy(y_pred, y))\n",
    "        self.log('val_precision', self.precision(y_pred, y))\n",
    "        self.log('val_recall', self.recall(y_pred, y))\n",
    "    \n",
    "    def configure_optimizers(self):\n",
    "        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)\n",
    "        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)\n",
    "        return {\n",
    "            'optimizer': optimizer,\n",
    "            'lr_scheduler': {\n",
    "                'scheduler': scheduler,\n",
    "                'interval': 'epoch',\n",
    "                'monitor': 'val_f1',\n",
    "                'frequency': 1,\n",
    "                'strict': True,\n",
    "            }\n",
    "        }"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
