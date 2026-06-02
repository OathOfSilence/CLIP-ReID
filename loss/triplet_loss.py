import torch
from torch import nn
import torch.distributed as dist


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n) #B, B
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist_mat = xx + yy
    dist_mat = dist_mat - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist_mat = dist_mat.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist_mat


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist_mat = xy_intersection/(x_norm * y_norm)
    dist_mat = (1. - dist_mat) / 2
    return dist_mat


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def _ranking_loss(self, dist_ap, dist_an):
        dist_ap = dist_ap * (1.0 + self.hard_factor)
        dist_an = dist_an * (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss

    def __call__(self, global_feat, labels, normalize_feature=False, **kwargs):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat) #B,B
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        loss = self._ranking_loss(dist_ap, dist_an)
        return loss, dist_ap, dist_an


class MemoryBank(object):
    """Momentum feature memory for global hard negative mining."""

    def __init__(self, momentum=0.2, normalize_feature=True):
        self.momentum = momentum
        self.normalize_feature = normalize_feature
        self.features = None
        self.labels = None
        self.initialized = None

    def _resize(self, num_entries, feat_dim, device):
        if self.features is not None and self.features.size(0) >= num_entries:
            if self.features.device != device:
                self.features = self.features.to(device)
                self.labels = self.labels.to(device)
                self.initialized = self.initialized.to(device)
            return

        old_size = 0 if self.features is None else self.features.size(0)
        new_features = torch.zeros(num_entries, feat_dim, device=device, dtype=torch.float32)
        new_labels = torch.full((num_entries,), -1, device=device, dtype=torch.long)
        new_initialized = torch.zeros(num_entries, device=device, dtype=torch.bool)

        if self.features is not None:
            new_features[:old_size].copy_(self.features.to(device))
            new_labels[:old_size].copy_(self.labels.to(device))
            new_initialized[:old_size].copy_(self.initialized.to(device))

        self.features = new_features
        self.labels = new_labels
        self.initialized = new_initialized

    def _gather_if_distributed(self, features, labels, indices):
        if not (dist.is_available() and dist.is_initialized()):
            return features, labels, indices

        gathered_features = [torch.zeros_like(features) for _ in range(dist.get_world_size())]
        gathered_labels = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
        gathered_indices = [torch.zeros_like(indices) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_features, features.contiguous())
        dist.all_gather(gathered_labels, labels.contiguous())
        dist.all_gather(gathered_indices, indices.contiguous())
        return torch.cat(gathered_features, 0), torch.cat(gathered_labels, 0), torch.cat(gathered_indices, 0)

    @torch.no_grad()
    def update(self, features, labels, indices):
        if indices is None:
            return

        features = features.detach().float()
        labels = labels.detach().long()
        indices = indices.detach().long()
        features, labels, indices = self._gather_if_distributed(features, labels, indices)

        if self.normalize_feature:
            features = normalize(features, axis=-1)

        max_index = int(indices.max().item()) + 1
        self._resize(max_index, features.size(1), features.device)

        initialized = self.initialized[indices]
        self.features[indices[~initialized]] = features[~initialized]
        if initialized.any():
            initialized_indices = indices[initialized]
            updated_features = self.momentum * self.features[initialized_indices] + (1.0 - self.momentum) * features[initialized]
            if self.normalize_feature:
                updated_features = normalize(updated_features, axis=-1)
            self.features[initialized_indices] = updated_features
        self.labels[indices] = labels
        self.initialized[indices] = True

    def get(self):
        if self.features is None:
            return None, None, None
        return self.features, self.labels, self.initialized


class MemoryTripletLoss(TripletLoss):
    """Triplet loss with batch hard positives and memory-bank hard negatives."""

    def __init__(self, margin=None, hard_factor=0.0, memory_momentum=0.2,
                 memory_topk_neg=1, memory_warmup_epochs=5, normalize_feature=True):
        super(MemoryTripletLoss, self).__init__(margin=margin, hard_factor=hard_factor)
        self.memory_bank = MemoryBank(momentum=memory_momentum, normalize_feature=normalize_feature)
        self.memory_topk_neg = max(1, int(memory_topk_neg))
        self.memory_warmup_epochs = int(memory_warmup_epochs)
        self.normalize_feature = normalize_feature

    def _memory_hard_negative(self, global_feat, labels, fallback_dist_an):
        memory_features, memory_labels, initialized = self.memory_bank.get()
        if memory_features is None or not initialized.any():
            return fallback_dist_an

        memory_features = memory_features[initialized]
        memory_labels = memory_labels[initialized]
        if memory_features.size(0) == 0:
            return fallback_dist_an

        dist_mat = euclidean_dist(global_feat, memory_features.detach())
        is_neg = labels.view(-1, 1).ne(memory_labels.view(1, -1))
        has_neg = is_neg.any(dim=1)
        dist_mat = dist_mat.masked_fill(~is_neg, float('inf'))

        max_candidates = dist_mat.size(1)
        topk = min(self.memory_topk_neg, max_candidates)
        if topk == 1:
            memory_dist_an, _ = torch.min(dist_mat, dim=1)
        else:
            topk_dist, _ = torch.topk(dist_mat, k=topk, dim=1, largest=False)
            random_column = torch.randint(topk, (dist_mat.size(0), 1), device=dist_mat.device)
            memory_dist_an = torch.gather(topk_dist, 1, random_column).squeeze(1)
            invalid_random = torch.isinf(memory_dist_an)
            if invalid_random.any():
                hardest_dist, _ = torch.min(dist_mat, dim=1)
                memory_dist_an = torch.where(invalid_random, hardest_dist, memory_dist_an)

        return torch.where(has_neg, memory_dist_an, fallback_dist_an)

    def __call__(self, global_feat, labels, indices=None, epoch=None, normalize_feature=False):
        should_normalize = normalize_feature or self.normalize_feature
        if should_normalize:
            global_feat = normalize(global_feat, axis=-1)

        dist_mat = euclidean_dist(global_feat, global_feat) #B,B
        dist_ap, batch_dist_an = hard_example_mining(dist_mat, labels)

        use_memory = indices is not None and epoch is not None and epoch > self.memory_warmup_epochs
        if use_memory:
            dist_an = self._memory_hard_negative(global_feat, labels, batch_dist_an)
        else:
            dist_an = batch_dist_an

        loss = self._ranking_loss(dist_ap, dist_an)
        self.memory_bank.update(global_feat, labels, indices)
        return loss, dist_ap, dist_an
